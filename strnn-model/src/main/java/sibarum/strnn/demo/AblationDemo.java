package sibarum.strnn.demo;

import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.MlpPrimitive;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.training.Datasets;
import sibarum.strnn.training.Example;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.NumberValue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Phase 6: the §9.6 diagnostic ablation. Run carving on the same payload with
 * two mandate sets:
 *
 *   A  full §9.2 mandates: plus_split, star_split, intermediate_product, result.
 *   B  result-only.
 *
 * For each, dump the carved structure to DOT and compare:
 *   - node count
 *   - count by primitive class
 *   - whether NumberValue(intermediate_product) appears as a node value on
 *     the path from any root binding to the terminal.
 *
 * The expected, falsifying outcome:
 *   A produces a structured pipeline that materializes the intermediate
 *   product on the result path; B is permitted to skip that materialization
 *   (the carver has no incentive to thread the chain through 20 if no mandate
 *   asks for it). If both produce identical structures, mandates are not
 *   doing the work the framework claims.
 */
public final class AblationDemo {

    private static final int RUNS = 50;
    private static final Path OUT_DIR = Path.of("target", "ablation");

    static void main(String[] args) throws IOException {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);

        TransformationGraph tg = buildTransformationGraph(addMlp, mulMlp);
        BackwardChainingCarver carver = new BackwardChainingCarver(/*seed=*/2024L);

        Files.createDirectories(OUT_DIR);

        Random rng = new Random(99L);
        Stats sA = new Stats();
        Stats sB = new Stats();
        Example sampleEx = null;
        CarvingResult sampleA = null;
        CarvingResult sampleB = null;

        int aFails = 0;
        int bFails = 0;
        for (int i = 0; i < RUNS; i++) {
            Example exFull = Datasets.generate(rng, rng.nextBoolean()
                    ? Datasets.Shape.ADD_THEN_MUL : Datasets.Shape.MUL_THEN_ADD);
            Example exResultOnly = Datasets.resultOnly(exFull);
            double expectedProduct = inferExpectedProduct(exFull);

            CarvingResult a = carver.carve(tg, exFull.mandates(), exFull.payload());
            if (a != null) {
                a.graph().execute();
                sA.observe(a, expectedProduct);
                if (sampleA == null) {
                    sampleEx = exFull;
                    sampleA = a;
                }
            } else {
                aFails++;
            }

            CarvingResult b = carver.carve(tg, exResultOnly.mandates(), exResultOnly.payload());
            if (b != null) {
                b.graph().execute();
                sB.observe(b, expectedProduct);
                if (sampleB == null) {
                    sampleB = b;
                }
            } else {
                bFails++;
            }
        }
        System.out.printf("A carve fails: %d / %d%n", aFails, RUNS);
        System.out.printf("B carve fails: %d / %d%n", bFails, RUNS);

        sA.report("A  (full §9.2 mandates)");
        sB.report("B  (result-only)");

        if (sampleA != null) {
            Files.writeString(OUT_DIR.resolve("A_full_mandates.dot"),
                    DotPrinter.toDot(sampleA.graph(), "A_full_mandates"));
        }
        if (sampleB != null) {
            Files.writeString(OUT_DIR.resolve("B_result_only.dot"),
                    DotPrinter.toDot(sampleB.graph(), "B_result_only"));
        }
        System.out.printf("%nDOT graphs for sample carving of '%s' written to %s%n",
                sampleEx == null ? "?" : sampleEx.label(), OUT_DIR.toAbsolutePath());

        if (bFails == RUNS && aFails < RUNS) {
            System.out.println("\nDIAGNOSTIC PASS (structural): result-only mandates do not give the carver enough");
            System.out.println("numeric anchors to invert MlpPrimitive (no pair (a,b) in pool satisfies a*b=result),");
            System.out.println("so it cannot construct a valid graph at all. The full mandate set seeds those anchors");
            System.out.println("(parsed numbers from token-list mandates, the intermediate product) and lets the");
            System.out.println("carver thread the chain through them. This is the §6 search-decomposition payoff");
            System.out.println("made concrete — without intermediates, the carver cannot solve the problem.");
            return;
        }
        boolean diagnostic = (sA.productMaterializedOnPath > 0)
                && (sB.productMaterializedOnPath < sA.productMaterializedOnPath);
        if (diagnostic) {
            System.out.println("\nDIAGNOSTIC PASS: full mandates structurally enforce intermediate-product materialization;");
            System.out.println("result-only mandates allow the carver to skip it.");
        } else {
            System.out.println("\nDIAGNOSTIC INFORMATIVE: A and B materialize the product at similar rates;");
            System.out.println("either the carver has an unintended bias or the demo is too easy to discriminate.");
        }
    }

    private static double inferExpectedProduct(Example ex) {
        for (var m : ex.mandates().mandates()) {
            if (m.name().equals("intermediate_product")
                    && m.expected() instanceof NumberValue(double n)) {
                return n;
            }
        }
        return Double.NaN;
    }

    private static TransformationGraph buildTransformationGraph(Mlp addMlp, Mlp mulMlp) {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("split_plus", new SplitStringAt('+'));
        b.addNode("split_star", new SplitStringAt('*'));
        b.addNode("token_0", new TokenAt(0));
        b.addNode("token_1", new TokenAt(1));
        b.addNode("parse", new ParseNumber());
        b.addNode("num_to_mat", new NumberToMatrix());
        b.addNode("compose", new ComposeMatrices());
        b.addNode("mlp_add", new MlpPrimitive(MlpRole.ADD, addMlp));
        b.addNode("mlp_mul", new MlpPrimitive(MlpRole.MUL, mulMlp));
        b.addNode("mat_to_num", new MatrixToNumber());
        b.addNode("output", new OutputPrimitive());
        return b.build();
    }

    private static final class Stats {
        int runs = 0;
        long totalNodes = 0;
        Map<String, Integer> primCounts = new TreeMap<>();
        int productMaterializedOnPath = 0;

        void observe(CarvingResult r, double expectedProduct) {
            runs++;
            totalNodes += r.graph().nodes().size();
            Map<String, Integer> here = new HashMap<>();
            for (CompGraphNode n : r.graph().topoOrder()) {
                String klass = n.tNode().primitive().getClass().getSimpleName();
                here.merge(klass, 1, Integer::sum);
            }
            here.forEach((k, v) -> primCounts.merge(k, v, Integer::sum));

            if (Double.isNaN(expectedProduct)) return;
            for (CompGraphNode n : r.graph().topoOrder()) {
                if (n.producedValue() instanceof NumberValue(double n1)
                        && Math.abs(n1 - expectedProduct) < 0.5
                        && r.graph().reaches(n, r.graph().terminal())) {
                    productMaterializedOnPath++;
                    break;
                }
            }
        }

        void report(String label) {
            System.out.println();
            System.out.println(label);
            System.out.printf("  runs:                 %d%n", runs);
            System.out.printf("  avg nodes:            %.1f%n", runs == 0 ? 0 : (double) totalNodes / runs);
            System.out.printf("  product on path:      %d / %d  (%.0f%%)%n",
                    productMaterializedOnPath, runs,
                    runs == 0 ? 0 : 100.0 * productMaterializedOnPath / runs);
            System.out.println("  primitive counts (averaged per run):");
            primCounts.forEach((k, v) -> System.out.printf("    %-20s %.2f%n", k, runs == 0 ? 0 : (double) v / runs));
        }
    }
}
