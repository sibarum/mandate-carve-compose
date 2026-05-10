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
 * doing the verifying. Runs the same pipeline twice — same primitives, same
 * mandate set, same scorers — under two trainer settings:
 *
 *   Run A (negative): axisLr = 0. Trainer has no objective for axis
 *     alignment. The framework's mandate verifier exposes the missing
 *     property as a FAIL on {@code axes_aligned}.
 *
 *   Run B (positive): axisLr {@literal >} 0. Trainer adds an axis-alignment
 *     objective that pulls dichotomies sharing a context atom toward parallel
 *     axes. The same mandate verifier confirms {@code axes_aligned} now passes.
 *
 * The contrast is the load-bearing demonstration: mandates encode testable
 * structural assertions, the trainer either supplies the matching objective
 * or doesn't, and the verifier reports honestly either way.
 */
public final class SemanticEmbeddingDemo {

    // Tolerance bands tuned to discriminate the "axis property holds" claim
    // from both directions:
    //   axes_aligned target +0.65 ± 0.30 → passes if axis_score ∈ [+0.35, +0.95].
    //   Run A's axis score (~0.147, near the d=32 random baseline of 0.141) is
    //   well below the band — FAIL is structurally correct.
    //   Run B's axis score (~0.82 after axis-alignment training) lands inside —
    //   PASS confirms the property is achievable when the trainer has the
    //   matching objective.
    private static final MandateSet MANDATES = new MandateSet(List.of(
            Mandate.intermediate("dichotomy_opposite",
                    new NumberValue(-0.60), 0.40, 0),
            Mandate.intermediate("context_clustered",
                    new NumberValue(0.15), 0.10, 1),
            Mandate.intermediate("axes_aligned",
                    new NumberValue(0.65), 0.30, 2),
            Mandate.result(new NumberValue(0.65), 0.30, 3)));

    public static void main(String[] args) throws IOException {
        String src = loadResource("/sample-semantics.txt");
        List<SemRelation> relations = SemanticParser.parseAll(src);
        System.out.printf(Locale.ROOT, "parsed %d relations, %d unique atoms%n%n",
                relations.size(), SemanticParser.collectAtoms(relations).size());

        int passA = runPass("Run A — no axis-alignment training (axisLr = 0)",
                relations, /*axisLr=*/0.0, /*seed=*/2024L);

        int passB = runPass("Run B — with axis-alignment training (axisLr = 0.01)",
                relations, /*axisLr=*/0.01, /*seed=*/2024L);

        System.out.println("\n========================================================");
        System.out.printf(Locale.ROOT,
                "comparison: Run A satisfied %d / %d mandates;%n"
                        + "            Run B satisfied %d / %d mandates.%n",
                passA, MANDATES.mandates().size(), passB, MANDATES.mandates().size());
        System.out.println("--------------------------------------------------------");
        System.out.println("same primitives, same scorers, same mandate set, same seed.");
        System.out.println("the only difference is the axisLr parameter on the trainer.");
        System.out.println("the framework's mandate machinery surfaces this as the");
        System.out.println("axes_aligned mandate moving from FAIL to PASS — exactly what");
        System.out.println("'mandates as testable structural assertions' should produce.");
    }

    private static int runPass(String label, List<SemRelation> relations, double axisLr, long seed) {
        System.out.println("========================================================");
        System.out.println(label);
        System.out.println("--------------------------------------------------------");

        int dim = 32;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, seed);
        double dichotomyLr = 0.05;
        double contextLr = 0.005;
        int epochs = 80;
        long t0 = System.nanoTime();
        SemanticTrainer.train(table, relations, dichotomyLr, contextLr, axisLr, epochs);
        long t1 = System.nanoTime();
        System.out.printf(Locale.ROOT,
                "trained %d epochs in %.2f s "
                        + "(dichotomyLr=%.3f, contextLr=%.4f, axisLr=%.4f)%n",
                epochs, (t1 - t0) / 1e9, dichotomyLr, contextLr, axisLr);

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

        double dScore = ((NumberValue) dNode.producedValue()).n();
        double cScore = ((NumberValue) cNode.producedValue()).n();
        double aScore = ((NumberValue) aNode.producedValue()).n();
        System.out.printf(Locale.ROOT, "  dichotomy_score = %+.4f%n", dScore);
        System.out.printf(Locale.ROOT, "  context_score   = %+.4f%n", cScore);
        System.out.printf(Locale.ROOT, "  axis_score      = %+.4f%n", aScore);

        VerificationReport report = new MandateVerifier().verify(cg, MANDATES);
        int satisfied = 0;
        System.out.println("  mandates:");
        for (var entry : report.outcomes().entrySet()) {
            Mandate m = entry.getKey();
            VerificationReport.Outcome o = entry.getValue();
            String tag = o.satisfied() ? "OK  " : "FAIL";
            if (o.satisfied()) satisfied++;
            String detail;
            if (o.satisfied()) {
                detail = String.format(Locale.ROOT,
                        "@%s, value %+.4f, target %+.2f ± %.2f",
                        o.producedBy().id(),
                        ((NumberValue) o.producedBy().producedValue()).n(),
                        ((NumberValue) m.expected()).n(),
                        m.tolerance());
            } else {
                detail = String.format(Locale.ROOT,
                        "target %+.2f ± %.2f — %s",
                        ((NumberValue) m.expected()).n(),
                        m.tolerance(),
                        o.reason());
            }
            System.out.printf(Locale.ROOT, "    %s %-22s %s%n", tag, m.name(), detail);
        }
        System.out.printf(Locale.ROOT, "  satisfied: %d / %d%n%n",
                satisfied, MANDATES.mandates().size());
        return satisfied;
    }

    private static String loadResource(String path) throws IOException {
        try (InputStream in = SemanticEmbeddingDemo.class.getResourceAsStream(path)) {
            if (in == null) throw new IOException("resource not found: " + path);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
