package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.training.InlineTrainer;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.List;
import java.util.Locale;

/**
 * Carver-assembled, inline-trained end-to-end demo. No hand-wired
 * ComputationGraph. The user supplies:
 *
 *   - A TransformationGraph populated with the KV primitives:
 *       embed-symbol  (StringValue   -> MatrixValue, Trainable)
 *       vector-transform (MatrixValue -> MatrixValue, Trainable bridge)
 *       lookup-symbol (MatrixValue   -> StringValue)
 *       string-output (StringValue   -> StringValue, terminal)
 *   - A root input atom ("hot").
 *   - A result mandate naming the expected output atom ("cold").
 *
 * The {@link BackwardChainingCarver} chains backward from the result mandate,
 * picks deterministic primitives where it can and the trainable bridge where
 * it must (the deterministic-only path from "hot" cannot reach "cold"; the
 * trainable bridge is the only edge in the substrate that can close the gap),
 * and returns a {@link CarvingResult} whose {@code simulatedValues} map names
 * the value each placed node is expected to produce.
 *
 * The generic {@link InlineTrainer} then walks that map. Every Trainable in
 * the carved graph receives its simulated value as its local training target.
 * Loop: forward / backward / step / verify. When the {@link sibarum.strnn.mandate.MandateVerifier}
 * reports PASS, the loop stops.
 *
 * Scope:
 *   - One Trainable in the gradient path of any given carving (the bridge).
 *     The embedding table is also Trainable but its target equals its current
 *     output, so its backward update is a no-op — gradients don't have to
 *     cross primitive boundaries.
 *   - No autograd-on-the-carving; that's the deferred multi-trainable step.
 *
 * What this demo demonstrates that earlier phases did not:
 *   - The ComputationGraph is built by the carver, not hand-wired.
 *   - The trainer is generic — it iterates over whatever trainables the
 *     carver placed, without knowing which primitive types they are.
 *   - The mandate {@code Mandate.result(StringValue("cold"))} simultaneously
 *     specifies (a) the structure the carver must produce, (b) the value
 *     the verifier checks, and (c) — via the carver's simulated values — the
 *     training target for every placed trainable.
 */
public final class CarverEndToEndDemo {

    public static void main(String[] args) {
        // ---- KV substrate ----
        int dim = 32;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, 42L);
        List<String> vocabulary = List.of(
                "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze");
        for (String s : vocabulary) table.embed(s);

        String inputAtom = "hot";
        String outputAtom = "cold";

        System.out.printf(Locale.ROOT, "vocabulary: %s%n", vocabulary);
        System.out.printf(Locale.ROOT, "root input:        '%s'%n", inputAtom);
        System.out.printf(Locale.ROOT, "mandate (result):  '%s'%n%n", outputAtom);

        // ---- Transformation graph (the substrate the carver searches) ----
        // any-to-any modulo type compatibility: the builder wires every
        // (output-type → input-type) edge it can.
        VectorTransform bridge = new VectorTransform(dim, dim, 7L);
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embed", new EmbedSymbol(table))
                .addNode("bridge", bridge)
                .addNode("lookup", new LookupSymbol(table))
                .addNode("output", new StringOutputPrimitive())
                .build();

        System.out.println("transformation graph edges (substrate):");
        tg.edges().forEach(e -> System.out.printf(Locale.ROOT,
                "  %-8s --> %s%n", e.from().id(), e.to().id()));
        System.out.println();

        // ---- Mandate ----
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(outputAtom), 0.0, 0)));

        // ---- Carve ----
        BackwardChainingCarver carver = new BackwardChainingCarver(123L);
        CarvingResult carving = carver.carve(tg, mandates, new StringValue(inputAtom));
        if (carving == null) {
            throw new AssertionError("carver failed to assemble a graph for the mandate");
        }

        System.out.println("carved computation graph:");
        printCarving(carving);
        System.out.println();

        // ---- Inline training ----
        InlineTrainer trainer = new InlineTrainer(
                carving, mandates,
                /* lr */ 0.1, /* maxSteps */ 500, /* checkEvery */ 25);

        System.out.println("inline training loop:");
        InlineTrainer.Result result = trainer.run();
        for (InlineTrainer.StepRecord rec : result.trace()) {
            System.out.printf(Locale.ROOT, "  step %3d: terminal='%s'  mandate:%s%n",
                    rec.step(),
                    ((StringValue) rec.terminalValue()).s(),
                    rec.mandatePass() ? "PASS" : "FAIL");
        }
        System.out.println();

        if (!result.converged()) {
            throw new AssertionError("inline training did not satisfy mandate within "
                    + result.stepsTaken() + " steps");
        }

        System.out.printf(Locale.ROOT,
                "Carver + InlineTrainer: mandate satisfied in %d step%s.%n"
                        + "No hand-wired computation graph; no per-trainable target derivation in user code.%n"
                        + "The carver assembled the path embed -> bridge -> lookup -> output from the%n"
                        + "TransformationGraph alone, picked the trainable bridge as the only edge that%n"
                        + "could close the gap between root='%s' and result='%s', and the generic trainer%n"
                        + "drove the bridge to a fixed-point that the MandateVerifier accepts.%n",
                result.stepsTaken(),
                result.stepsTaken() == 1 ? "" : "s",
                inputAtom, outputAtom);
    }

    private static void printCarving(CarvingResult carving) {
        for (CompGraphNode n : carving.graph().topoOrder()) {
            StringBuilder sb = new StringBuilder();
            sb.append("  ").append(n.id())
                    .append(" [").append(n.tNode().primitive().name()).append("]");
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource src = n.slot(i);
                if (src == null) {
                    sb.append(" slot").append(i).append("<-ROOT");
                } else {
                    sb.append(" slot").append(i).append("<-").append(src.source().id());
                }
            }
            Value sim = carving.simulatedValues().get(n);
            if (sim != null) {
                sb.append("  simulated=").append(summarize(sim));
            }
            System.out.println(sb);
        }
    }

    private static String summarize(Value v) {
        return switch (v) {
            case StringValue sv -> "'" + sv.s() + "'";
            case sibarum.strnn.value.MatrixValue mv -> "matrix[dim=" + mv.dim() + "]";
            case sibarum.strnn.value.NumberValue nv -> Double.toString(nv.n());
            default -> v.toString();
        };
    }
}
