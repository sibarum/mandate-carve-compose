package sibarum.strnn.demo;

import sibarum.strnn.cache.CosineSimilarity;
import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.NumberSub;
import sibarum.strnn.cache.SymbolEmbeddingTable;
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
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Emergent similarity-based routing in a network.
 *
 * The same network — {@code embed → [cos_spatial, cos_motion] → diff → output}
 * — is run on every test atom under two substrate conditions and verified
 * with two mandate placements:
 *
 *   Run A: untrained substrate. Random embeddings. The network is correctly
 *     assembled and the references are computed as group centroids, but the
 *     vector geometry doesn't carry meaning. Routing collapses to chance.
 *
 *   Run B: trained substrate. Same network, same primitives, same wiring.
 *     The trained geometry organizes atoms by semantic group, so cos sim
 *     to the right centroid is high and to the wrong one is low. Routing
 *     accuracy goes high.
 *
 * Two mandate placements per atom:
 *
 *   Case A — mandate the split. The mandate names the gate score
 *     directly: cos_spatial in the expected direction.
 *
 *   Case B — mandate downstream. The mandate names the terminal value
 *     (the difference of similarities). The split is not named, but it
 *     must be happening for the downstream mandate to be satisfied.
 *
 * The 2×2 grid (untrained/trained × split/downstream) is the demonstration:
 * the split happens in both cases regardless of where the mandate is placed,
 * but only when the substrate's geometry supports it.
 */
public final class SimilarityRoutingDemo {

    // Two coherent clusters — every member of each group shares at least one
    // rhs atom with every other member, so context-pull training puts them in
    // the same neighborhood:
    //   ORIENTATION group: all share rhs "orientation"
    //     (top|bottom), (parallel|perpendicular), (straight|twisted)
    //   MOTION group: all share rhs "motion"
    //     (fast|slow), (dynamic|static), (active|sedentary)
    // Only the positive half of each dichotomy is included to avoid antipodal
    // cancellation in the centroid post-training.
    private static final List<String> ORIENTATION = List.of(
            "top", "parallel", "straight");
    private static final List<String> MOTION = List.of(
            "fast", "dynamic", "active");
    private static final int DIM = 32;
    private static final long SEED = 2024L;

    public static void main(String[] args) throws IOException {
        String src = loadResource("/sample-semantics.txt");
        List<SemRelation> relations = SemanticParser.parseAll(src);
        System.out.printf(Locale.ROOT,
                "parsed %d relations, %d unique atoms%n%n",
                relations.size(), SemanticParser.collectAtoms(relations).size());

        PassResult untrained = runPass(
                "Run A — untrained substrate (random embeddings)",
                relations, /*train=*/false);

        PassResult trained = runPass(
                "Run B — trained substrate (full multi-objective)",
                relations, /*train=*/true);

        printComparison(untrained, trained);
    }

    // ---------------------------------------------------------------------

    private static PassResult runPass(String label, List<SemRelation> relations, boolean train) {
        System.out.println("========================================================");
        System.out.println(label);
        System.out.println("--------------------------------------------------------");

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(DIM, SEED);
        if (train) {
            // Pure context-pull training for the routing demo. Dichotomy push
            // is OFF: it forces paired atoms antipodal, which drives every
            // rhs context atom toward the origin (it gets pulled equally by
            // both antipodal halves of each dichotomy it appears in). With
            // origin-anchored centroids, group-vs-group separation collapses.
            // Context pull alone clusters atoms by shared rhs without that
            // pathology — exactly what cluster-based routing needs.
            SemanticTrainer.train(table, relations,
                    /*dichotomyLr=*/0.0,
                    /*contextLr=*/0.05,
                    /*axisLr=*/0.0,
                    /*epochs=*/100);
            System.out.println("trained 100 epochs (context pull only — no antipodal forcing)");
        } else {
            // Force lazy init for every atom so untrained = random init for the same atom set.
            for (SemRelation r : relations) {
                table.embed(r.lhs().left());
                table.embed(r.lhs().right());
            }
            for (String s : ORIENTATION) table.embed(s);
            for (String s : MOTION) table.embed(s);
            System.out.println("no training (random embeddings)");
        }

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embed", new EmbedSymbol(table))
                .addNode("cos_spatial", new CosineSimilarity())
                .addNode("cos_motion", new CosineSimilarity())
                .addNode("diff", new NumberSub())
                .addNode("output", new OutputPrimitive())
                .build();

        int caseAPassed = 0;
        int caseBPassed = 0;
        int total = 0;

        System.out.println("per-atom routing (leave-one-out centroids):");
        List<String> allTests = new ArrayList<>();
        allTests.addAll(ORIENTATION);
        allTests.addAll(MOTION);

        for (String atom : allTests) {
            boolean expectSpatial = ORIENTATION.contains(atom);
            // Leave-one-out: the spatial centroid excludes the test atom when the
            // test atom is spatial; same for motion. This removes the
            // self-inclusion bias that lets random embeddings classify their own
            // contributors correctly.
            MatrixValue refSpatial = new MatrixValue(
                    centroidExcluding(table, ORIENTATION, expectSpatial ? atom : null));
            MatrixValue refMotion = new MatrixValue(
                    centroidExcluding(table, MOTION, expectSpatial ? null : atom));

            AtomResult r = runOneAtom(tg, atom, refSpatial, refMotion, expectSpatial);

            System.out.printf(Locale.ROOT,
                    "  %-10s [%s]  cos_o=%+.3f cos_m=%+.3f diff=%+.3f   caseA:%s   caseB:%s%n",
                    atom,
                    expectSpatial ? "orient" : "motion",
                    r.cosSpatial, r.cosMotion, r.diff,
                    r.caseAPass ? "PASS" : "FAIL",
                    r.caseBPass ? "PASS" : "FAIL");

            if (r.caseAPass) caseAPassed++;
            if (r.caseBPass) caseBPassed++;
            total++;
        }

        System.out.printf(Locale.ROOT, "Case A (mandate the split):     %d / %d atoms%n",
                caseAPassed, total);
        System.out.printf(Locale.ROOT, "Case B (mandate downstream):    %d / %d atoms%n%n",
                caseBPassed, total);

        return new PassResult(caseAPassed, caseBPassed, total);
    }

    // ---------------------------------------------------------------------

    private static AtomResult runOneAtom(TransformationGraph tg,
                                         String atom,
                                         MatrixValue refSpatial,
                                         MatrixValue refMotion,
                                         boolean expectSpatial) {
        CompGraphNode embed = new CompGraphNode("embed_" + atom, tg.node("embed"));
        CompGraphNode cosS = new CompGraphNode("cosS_" + atom, tg.node("cos_spatial"));
        CompGraphNode cosM = new CompGraphNode("cosM_" + atom, tg.node("cos_motion"));
        CompGraphNode diffN = new CompGraphNode("diff_" + atom, tg.node("diff"));
        CompGraphNode out = new CompGraphNode("out_" + atom, tg.node("output"));

        cosS.wire(0, new SlotSource(embed, tg.edge(tg.node("embed"), tg.node("cos_spatial"))));
        cosM.wire(0, new SlotSource(embed, tg.edge(tg.node("embed"), tg.node("cos_motion"))));
        diffN.wire(0, new SlotSource(cosS, tg.edge(tg.node("cos_spatial"), tg.node("diff"))));
        diffN.wire(1, new SlotSource(cosM, tg.edge(tg.node("cos_motion"), tg.node("diff"))));
        out.wire(0, new SlotSource(diffN, tg.edge(tg.node("diff"), tg.node("output"))));

        ComputationGraph cg = new ComputationGraph(
                List.of(embed, cosS, cosM, diffN, out), out);
        cg.bindRoot(embed, 0, new StringValue(atom));
        cg.bindRoot(cosS, 1, refSpatial);
        cg.bindRoot(cosM, 1, refMotion);
        cg.execute();

        double cosSpatial = ((NumberValue) cosS.producedValue()).n();
        double cosMotion = ((NumberValue) cosM.producedValue()).n();
        double diff = ((NumberValue) diffN.producedValue()).n();

        // Tight bands so mandates discriminate meaningfully:
        //   Case A target +0.85 ± 0.15 → band [0.70, 1.00].
        //     Trained own-cluster cosines land near +0.95, comfortably inside.
        //     Trained other-cluster cosines land near +0.40, outside.
        //     Untrained random cosines mostly outside (band covers only 15%
        //     of the [-1, +1] range), failing for most random configurations.
        //   Case B target ±0.40 ± 0.30 → band [±0.10, ±0.70].
        //     Trained diffs land in this band; untrained diffs cluster near
        //     zero and miss it.
        NumberValue caseAHigh = new NumberValue(+0.85);
        double caseATol = 0.15;
        NumberValue caseBPos = new NumberValue(+0.40);
        NumberValue caseBNeg = new NumberValue(-0.40);
        double caseBTol = 0.30;

        MandateSet caseA;
        MandateSet caseB;
        if (expectSpatial) {
            caseA = new MandateSet(List.of(
                    Mandate.intermediate("cos_orient_high", caseAHigh, caseATol, 0)));
            caseB = new MandateSet(List.of(
                    Mandate.result(caseBPos, caseBTol, 0)));
        } else {
            caseA = new MandateSet(List.of(
                    Mandate.intermediate("cos_motion_high", caseAHigh, caseATol, 0)));
            caseB = new MandateSet(List.of(
                    Mandate.result(caseBNeg, caseBTol, 0)));
        }

        VerificationReport reportA = new MandateVerifier().verify(cg, caseA);
        VerificationReport reportB = new MandateVerifier().verify(cg, caseB);

        return new AtomResult(cosSpatial, cosMotion, diff,
                reportA.allSatisfied(), reportB.allSatisfied());
    }

    // ---------------------------------------------------------------------

    private static double[] centroidExcluding(SymbolEmbeddingTable table,
                                               List<String> atoms,
                                               String excluded) {
        double[] sum = new double[table.dim()];
        int n = 0;
        for (String s : atoms) {
            if (s.equals(excluded)) continue;
            double[] e = table.embed(s);
            for (int i = 0; i < sum.length; i++) sum[i] += e[i];
            n++;
        }
        if (n == 0) {
            throw new IllegalStateException("centroid would be over zero atoms (group too small)");
        }
        for (int i = 0; i < sum.length; i++) sum[i] /= n;
        return sum;
    }

    private static void printComparison(PassResult untrained, PassResult trained) {
        int total = untrained.total;
        System.out.println("========================================================");
        System.out.println("Comparison: 2×2 grid");
        System.out.println("--------------------------------------------------------");
        System.out.printf(Locale.ROOT, "                    Case A (split)   Case B (downstream)%n");
        System.out.printf(Locale.ROOT, "  Untrained:           %2d / %d              %2d / %d%n",
                untrained.caseA, total, untrained.caseB, total);
        System.out.printf(Locale.ROOT, "  Trained:             %2d / %d              %2d / %d%n",
                trained.caseA, total, trained.caseB, total);
        System.out.println("--------------------------------------------------------");
        System.out.println("same network structure (embed → cos_s, cos_m → diff → output),");
        System.out.println("same primitives, same wiring, same mandate forms — only the");
        System.out.println("substrate's training state changes between Run A and Run B.");
        System.out.println();
        System.out.println("Case A and Case B verify the same routing decision through");
        System.out.println("different network positions: Case A reads the gate score");
        System.out.println("directly (the split); Case B reads the terminal value (downstream");
        System.out.println("of the split). When the routing emerges, both pass; when the");
        System.out.println("substrate can't carry it, both fail. The split happens in both");
        System.out.println("cases regardless of mandate placement, when the geometry supports it.");
    }

    private static String loadResource(String path) throws IOException {
        try (InputStream in = SimilarityRoutingDemo.class.getResourceAsStream(path)) {
            if (in == null) throw new IOException("resource not found: " + path);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    // ---------------------------------------------------------------------

    private record AtomResult(double cosSpatial, double cosMotion, double diff,
                              boolean caseAPass, boolean caseBPass) {
    }

    private record PassResult(int caseA, int caseB, int total) {
    }
}
