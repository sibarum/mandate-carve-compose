package sibarum.strnn.demo;

import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.cache.semantic.AxisAlignmentScorer;
import sibarum.strnn.cache.semantic.ContextClusterScorer;
import sibarum.strnn.cache.semantic.DichotomyOppositionScorer;
import sibarum.strnn.cache.semantic.SemRelation;
import sibarum.strnn.cache.semantic.SemanticParser;
import sibarum.strnn.cache.semantic.SemanticTrainer;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.NumberValue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;

/**
 * One composition for the KV-cache semantic-embedding work, with mandates
 * doing the verifying.
 *
 * Pipeline:
 *   1. Parse {@code sample-semantics.txt}.
 *   2. Train the embedding table multi-objectively (push dichotomy pairs
 *      apart, pull rhs context atoms together).
 *   3. Build a single ComputationGraph wiring three scoring primitives in
 *      sequence — dichotomy → context → axis → output. Each scorer reads
 *      the trained table and produces a NumberValue score.
 *   4. Define a MandateSet with three intermediate mandates (one per
 *      structural property) and one result mandate (on the terminal value).
 *   5. {@link MandateVerifier} checks the full set in one pass.
 *
 * The framework's claim made concrete: structural requirements expressed as
 * mandates, one carved (here, manually wired) computation, the verifier
 * reports which mandates each node satisfies.
 */
public final class SemanticEmbeddingDemo {

    public static void main(String[] args) throws IOException {
        // ---- 1. Parse ----
        String src = loadResource("/sample-semantics.txt");
        List<SemRelation> relations = SemanticParser.parseAll(src);
        System.out.printf(Locale.ROOT, "parsed %d relations, %d unique atoms%n",
                relations.size(), SemanticParser.collectAtoms(relations).size());

        // ---- 2. Train ----
        int dim = 32;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, 2024L);
        double dichotomyLr = 0.05;
        double contextLr = 0.005;
        int epochs = 80;
        long t0 = System.nanoTime();
        SemanticTrainer.train(table, relations, dichotomyLr, contextLr, epochs);
        long t1 = System.nanoTime();
        System.out.printf(Locale.ROOT, "trained %d epochs in %.2f s (dichotomyLr=%.3f, contextLr=%.4f)%n",
                epochs, (t1 - t0) / 1e9, dichotomyLr, contextLr);
        System.out.printf(Locale.ROOT, "table size after training: %d symbols at dim %d%n",
                table.size(), dim);

        // ---- 3. Build the composition: dichotomy → context → axis → output ----
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("dichotomy_score", new DichotomyOppositionScorer(table, relations))
                .addNode("context_score", new ContextClusterScorer(table, relations))
                .addNode("axis_score", new AxisAlignmentScorer(table, relations))
                .addNode("output", new OutputPrimitive())
                .build();

        CompGraphNode dNode = new CompGraphNode("c0_dichotomy", tg.node("dichotomy_score"));
        CompGraphNode cNode = new CompGraphNode("c1_context", tg.node("context_score"));
        CompGraphNode aNode = new CompGraphNode("c2_axis", tg.node("axis_score"));
        CompGraphNode oNode = new CompGraphNode("c3_output", tg.node("output"));

        TransformationEdge dToC = tg.edge(tg.node("dichotomy_score"), tg.node("context_score"));
        TransformationEdge cToA = tg.edge(tg.node("context_score"), tg.node("axis_score"));
        TransformationEdge aToO = tg.edge(tg.node("axis_score"), tg.node("output"));

        cNode.wire(0, new SlotSource(dNode, dToC));
        aNode.wire(0, new SlotSource(cNode, cToA));
        oNode.wire(0, new SlotSource(aNode, aToO));

        ComputationGraph cg = new ComputationGraph(List.of(dNode, cNode, aNode, oNode), oNode);
        cg.bindRoot(dNode, 0, new NumberValue(0.0));
        cg.execute();

        // ---- Surface the actual scores so we can read them. ----
        double dScore = ((NumberValue) dNode.producedValue()).n();
        double cScore = ((NumberValue) cNode.producedValue()).n();
        double aScore = ((NumberValue) aNode.producedValue()).n();
        double terminalScore = ((NumberValue) oNode.producedValue()).n();
        System.out.printf(Locale.ROOT, "%nproduced scores:%n");
        System.out.printf(Locale.ROOT, "  dichotomy_score (avg cos of dichotomy pairs)  = %+.4f  (target: near −1)%n", dScore);
        System.out.printf(Locale.ROOT, "  context_score   (within − between cosine)     = %+.4f  (target: > 0)%n", cScore);
        System.out.printf(Locale.ROOT, "  axis_score      (avg |cos| of shared axes)    = %+.4f  (target: > 0)%n", aScore);
        System.out.printf(Locale.ROOT, "  terminal value (passed through from axis)     = %+.4f%n", terminalScore);

        // ---- 4. Define mandates ----
        // Tolerances are tight enough to encode meaningful structural claims, not just
        // "any score in a wide band." Each mandate names a specific bar:
        //   dichotomy_opposite: cos far below 0 (well past the 0 random baseline)
        //   context_clustered:  positive within−between margin
        //   axes_aligned:       avg |cos| substantially above the random baseline
        //                       (~0.141 at d=32; bar set at 0.30)
        // The result mandate echoes the axis claim because the terminal passes that score
        // through. Mandates that do not hold will fail honestly — the framework's job.
        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("dichotomy_opposite",
                        new NumberValue(-0.70), 0.30, 0),
                Mandate.intermediate("context_clustered",
                        new NumberValue(0.15), 0.10, 1),
                Mandate.intermediate("axes_aligned",
                        new NumberValue(0.40), 0.10, 2),
                Mandate.result(new NumberValue(0.40), 0.10, 3)));

        // ---- 5. Verify ----
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        System.out.println("\nmandate verification:");
        int pass = 0;
        int fail = 0;
        for (var entry : report.outcomes().entrySet()) {
            Mandate m = entry.getKey();
            VerificationReport.Outcome o = entry.getValue();
            String tag = o.satisfied() ? "OK  " : "FAIL";
            if (o.satisfied()) pass++;
            else fail++;
            String where = o.satisfied()
                    ? String.format(Locale.ROOT, "@%s, value %+.4f, target %+.2f ± %.2f",
                            o.producedBy().id(),
                            ((NumberValue) o.producedBy().producedValue()).n(),
                            ((NumberValue) m.expected()).n(),
                            m.tolerance())
                    : String.format(Locale.ROOT, "target %+.2f ± %.2f — %s",
                            ((NumberValue) m.expected()).n(),
                            m.tolerance(),
                            o.reason());
            System.out.printf(Locale.ROOT, "  %s %-22s %s%n", tag, m.name(), where);
        }

        // Bottom line: structural claims that hold vs. structural claims the framework exposes as not-yet-met.
        System.out.printf(Locale.ROOT, "%nstructural claims: %d satisfied / %d total%n", pass, pass + fail);
        if (fail > 0) {
            System.out.println("the framework correctly surfaces unmet claims — they're real signals,");
            System.out.println("not failures of the demo. Each FAIL identifies a property the trainer");
            System.out.println("does not yet have an objective for.");
        }
        System.out.printf(Locale.ROOT, "%nKV-cache semantic embedding: one composition, %d mandates evaluated.%n",
                mandates.mandates().size());
    }

    private static String loadResource(String path) throws IOException {
        try (InputStream in = SemanticEmbeddingDemo.class.getResourceAsStream(path)) {
            if (in == null) throw new IOException("resource not found: " + path);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
