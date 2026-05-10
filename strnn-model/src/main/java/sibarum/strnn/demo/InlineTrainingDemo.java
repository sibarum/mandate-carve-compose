package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;

import java.util.List;
import java.util.Locale;

/**
 * Inline KV training driven by mandates.
 *
 * No pre-training. A single mandate specifies what should flow in (an atom)
 * and what should come out (another atom). The carved graph is:
 *
 *   StringValue(A) → EmbedSymbol → VectorTransform → LookupSymbol → StringValue(B)
 *
 * The VectorTransform is the trainable bridge. Its weights start random;
 * the embedding table is initialized lazily for every atom in the vocabulary.
 * The training loop drives the bridge to map {@code embed(A)} toward
 * {@code embed(B)}, supervised by the mandate's expected output.
 *
 * The loop:
 *   1. Forward — execute the whole graph; the terminal emits whatever atom
 *      the bridge's current state currently points at (initially random).
 *   2. Backward — the mandate's expected output is the training target;
 *      VectorTransform.backward(embed(B)) computes ∂L/∂W under squared error.
 *   3. Step — apply SGD to W.
 *   4. Repeat until the terminal output matches the mandate's expected
 *      output, or budget is exhausted.
 *
 * After convergence, the same MandateVerifier that v2 / phase 1 / phase 2
 * used confirms the terminal mandate. The training is *inside* the
 * mandate-driven loop; the graph and mandates don't change between
 * iterations — only the trainable's parameters do.
 *
 * What this demonstrates:
 *   - A mandate (input + output) can serve simultaneously as the carving
 *     spec and the training signal.
 *   - The carved graph hosts the training loop directly — no separate
 *     pre-training step required.
 *   - Static primitives (EmbedSymbol's table once initialized, LookupSymbol,
 *     terminal) coexist in the same graph with the trainable bridge,
 *     untouched by training.
 */
public final class InlineTrainingDemo {

    public static void main(String[] args) {
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(32, 42L);

        // A small vocabulary so LookupSymbol has real competitors to choose from.
        // The trainable bridge needs to drive its output close enough to embed("cold")
        // that "cold" wins the nearest-neighbour test against everything else here.
        List<String> vocabulary = List.of(
                "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze");
        for (String s : vocabulary) table.embed(s);

        String inputAtom = "hot";
        String outputAtom = "cold";
        System.out.printf(Locale.ROOT,
                "vocabulary: %s%n", vocabulary);
        System.out.printf(Locale.ROOT,
                "mandate: input='%s'  →  output='%s'%n%n", inputAtom, outputAtom);

        // ---- Build the carved graph ----
        VectorTransform bridge = new VectorTransform(32, 32, 7L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embed", new EmbedSymbol(table))
                .addNode("bridge", bridge)
                .addNode("lookup", new LookupSymbol(table))
                .addNode("output", new StringOutputPrimitive())
                .build();

        CompGraphNode embedN = new CompGraphNode("c0_embed", tg.node("embed"));
        CompGraphNode bridgeN = new CompGraphNode("c1_bridge", tg.node("bridge"));
        CompGraphNode lookupN = new CompGraphNode("c2_lookup", tg.node("lookup"));
        CompGraphNode outN = new CompGraphNode("c3_output", tg.node("output"));

        TransformationEdge eToB = tg.edge(tg.node("embed"), tg.node("bridge"));
        TransformationEdge bToL = tg.edge(tg.node("bridge"), tg.node("lookup"));
        TransformationEdge lToO = tg.edge(tg.node("lookup"), tg.node("output"));

        bridgeN.wire(0, new SlotSource(embedN, eToB));
        lookupN.wire(0, new SlotSource(bridgeN, bToL));
        outN.wire(0, new SlotSource(lookupN, lToO));

        ComputationGraph cg = new ComputationGraph(
                List.of(embedN, bridgeN, lookupN, outN), outN);
        cg.bindRoot(embedN, 0, new StringValue(inputAtom));

        // ---- Initial state (before training) ----
        cg.execute();
        String initialOutput = ((StringValue) outN.producedValue()).s();
        System.out.printf(Locale.ROOT, "before training:%n  terminal = '%s'%n%n", initialOutput);

        // ---- Mandate ----
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(outputAtom), 0.0, 0)));

        VerificationReport initialReport = new MandateVerifier().verify(cg, mandates);
        System.out.printf(Locale.ROOT, "initial mandate verification: %s%n%n",
                initialReport.allSatisfied() ? "PASS" : "FAIL");

        // ---- Inline training loop ----
        // Target for the trainable bridge: the embedding of the mandate's expected output.
        // The bridge optimizes to map embed(input) → embed(output) under squared loss.
        // The graph and mandate are unchanged between iterations; only the bridge updates.
        double lr = 0.1;
        int maxSteps = 500;
        int checkEvery = 25;
        MatrixValue trainingTarget = new MatrixValue(table.embed(outputAtom));

        System.out.println("inline training loop (mandate-driven, terminal: forward → backward → step):");
        int step = 0;
        boolean converged = false;
        for (step = 1; step <= maxSteps; step++) {
            cg.execute();
            bridge.backward(trainingTarget);
            bridge.step(lr);

            if (step % checkEvery == 0 || step == 1) {
                cg.execute();
                String currentOutput = ((StringValue) outN.producedValue()).s();
                VerificationReport rep = new MandateVerifier().verify(cg, mandates);
                System.out.printf(Locale.ROOT, "  step %3d: terminal='%s'  mandate:%s%n",
                        step, currentOutput, rep.allSatisfied() ? "PASS" : "FAIL");
                if (rep.allSatisfied()) {
                    converged = true;
                    break;
                }
            }
        }

        if (!converged) {
            // Final check after budget exhausted.
            cg.execute();
        }

        // ---- Final verification ----
        cg.execute();
        String finalOutput = ((StringValue) outN.producedValue()).s();
        VerificationReport finalReport = new MandateVerifier().verify(cg, mandates);
        System.out.printf(Locale.ROOT, "%nafter training:%n  terminal = '%s'%n", finalOutput);
        System.out.printf(Locale.ROOT, "  final mandate verification: %s%n",
                finalReport.allSatisfied() ? "PASS" : "FAIL");

        if (!finalReport.allSatisfied()) {
            throw new AssertionError(
                    "inline training did not satisfy mandate after " + step + " steps; "
                            + "final terminal = '" + finalOutput + "'");
        }

        System.out.printf(Locale.ROOT,
                "%nInline training: mandate satisfied in %d step%s. The bridge's W%n"
                        + "was tuned from random init to map embed(%s) toward embed(%s),%n"
                        + "with the mandate serving as both the structural spec and the%n"
                        + "training target. No pre-training; no separate trainer call;%n"
                        + "training happens inside the carved graph's forward/backward loop.%n",
                step, step == 1 ? "" : "s", inputAtom, outputAtom);
    }
}
