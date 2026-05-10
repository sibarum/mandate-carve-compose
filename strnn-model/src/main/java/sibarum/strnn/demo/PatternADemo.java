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
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.primitive.TransformerPrimitive;
import sibarum.strnn.training.Datasets;
import sibarum.strnn.training.Example;
import sibarum.strnn.transformer.Transformer;
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
 * Pattern A from the v1 plan: drop-in swap of the underlying learned component.
 * Builds two transformation graphs with the same structure but different
 * trainable primitives (MLP vs. Transformer for the ADD/MUL roles), runs the
 * §9.6 ablation on each, and prints a side-by-side comparison so we can see
 * whether the carved orchestration depends on which learner sits inside the
 * MlpPrimitive/TransformerPrimitive slots.
 *
 * If A and the transformer-A produce structurally similar carvings, the
 * orchestration claim survives a real architecture swap. If they diverge,
 * we learn which assumptions in the carver were MLP-specific.
 */
public final class PatternADemo {
    private static final int RUNS = 50;
    private static final Path OUT_DIR = Path.of("target", "pattern-a");

    public static void main(String[] args) throws IOException {
        Files.createDirectories(OUT_DIR);

        System.out.println("=== Configuration MLP ===");
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);
        TransformationGraph mlpTg = buildGraphMlp(addMlp, mulMlp);
        Stats mlpA = new Stats();
        Stats mlpB = new Stats();
        CarvingResult mlpSampleA = runAblation(mlpTg, mlpA, mlpB, /*seed=*/2024L);

        System.out.println("\n=== Configuration TRANSFORMER ===");
        Transformer addTfm = new Transformer(2, 1, 16, 64, 1, 42L);
        Transformer mulTfm = new Transformer(2, 1, 32, 128, 1, 1337L);
        TrainingDemoBootstrap.pretrainTransformerSilent(addTfm, MlpRole.ADD, 4000, 32, 0.005);
        TrainingDemoBootstrap.pretrainTransformerSilent(mulTfm, MlpRole.MUL, 20000, 32, 0.0015);
        TransformationGraph tfmTg = buildGraphTransformer(addTfm, mulTfm);
        Stats tfmA = new Stats();
        Stats tfmB = new Stats();
        CarvingResult tfmSampleA = runAblation(tfmTg, tfmA, tfmB, /*seed=*/2024L);

        System.out.println("\n=== Side-by-side ===");
        System.out.printf("%-30s %-15s %-15s%n", "metric", "MLP", "Transformer");
        System.out.printf("%-30s %-15.1f %-15.1f%n", "A: avg nodes",
                mlpA.runs == 0 ? 0 : (double) mlpA.totalNodes / mlpA.runs,
                tfmA.runs == 0 ? 0 : (double) tfmA.totalNodes / tfmA.runs);
        System.out.printf("%-30s %-15s %-15s%n", "A: success / runs",
                mlpA.runs + " / " + RUNS, tfmA.runs + " / " + RUNS);
        System.out.printf("%-30s %-15.0f %-15.0f%n", "A: % product on path",
                mlpA.runs == 0 ? 0 : 100.0 * mlpA.productOnPath / mlpA.runs,
                tfmA.runs == 0 ? 0 : 100.0 * tfmA.productOnPath / tfmA.runs);
        System.out.printf("%-30s %-15s %-15s%n", "B: success / runs",
                mlpB.runs + " / " + RUNS, tfmB.runs + " / " + RUNS);

        System.out.println("\nA primitive-class counts (averaged per run):");
        printSideBySide(mlpA.primCounts, tfmA.primCounts, mlpA.runs, tfmA.runs);

        if (mlpSampleA != null) {
            Files.writeString(OUT_DIR.resolve("A_mlp.dot"),
                    DotPrinter.toDot(mlpSampleA.graph(), "A_mlp"));
        }
        if (tfmSampleA != null) {
            Files.writeString(OUT_DIR.resolve("A_transformer.dot"),
                    DotPrinter.toDot(tfmSampleA.graph(), "A_transformer"));
        }
        System.out.printf("%nDOT graphs written to %s%n", OUT_DIR.toAbsolutePath());

        boolean structurallyEquivalent = primCountsMatch(mlpA, tfmA)
                && Math.abs(mlpA.totalNodes - tfmA.totalNodes) <= mlpA.runs;
        if (structurallyEquivalent) {
            System.out.println("\nPATTERN A PASS: carved orchestration is structurally equivalent");
            System.out.println("under both architectures. The framework's component-agnostic claim");
            System.out.println("survives this swap test.");
        } else {
            System.out.println("\nPATTERN A INFORMATIVE: carved orchestration differs between MLP and");
            System.out.println("Transformer configurations. Inspect primitive-class counts above to");
            System.out.println("see which choices the carver made differently.");
        }
    }

    private static CarvingResult runAblation(TransformationGraph tg, Stats sA, Stats sB, long seed) {
        BackwardChainingCarver carver = new BackwardChainingCarver(seed);
        Random rng = new Random(99L);
        CarvingResult sampleA = null;
        for (int i = 0; i < RUNS; i++) {
            Example exFull = Datasets.generate(rng, rng.nextBoolean()
                    ? Datasets.Shape.ADD_THEN_MUL : Datasets.Shape.MUL_THEN_ADD);
            Example exResultOnly = Datasets.resultOnly(exFull);
            double expectedProduct = inferExpectedProduct(exFull);

            CarvingResult a = carver.carve(tg, exFull.mandates(), exFull.payload());
            if (a != null) {
                a.graph().execute();
                sA.observe(a, expectedProduct);
                if (sampleA == null) sampleA = a;
            }
            CarvingResult b = carver.carve(tg, exResultOnly.mandates(), exResultOnly.payload());
            if (b != null) {
                b.graph().execute();
                sB.observe(b, expectedProduct);
            }
        }
        return sampleA;
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

    private static TransformationGraph buildGraphMlp(Mlp addMlp, Mlp mulMlp) {
        return baseBuilder()
                .addNode("mlp_add", new MlpPrimitive(MlpRole.ADD, addMlp))
                .addNode("mlp_mul", new MlpPrimitive(MlpRole.MUL, mulMlp))
                .build();
    }

    private static TransformationGraph buildGraphTransformer(Transformer addTfm, Transformer mulTfm) {
        return baseBuilder()
                .addNode("tfm_add", new TransformerPrimitive(MlpRole.ADD, addTfm))
                .addNode("tfm_mul", new TransformerPrimitive(MlpRole.MUL, mulTfm))
                .build();
    }

    private static TransformationGraphBuilder baseBuilder() {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("split_plus", new SplitStringAt('+'));
        b.addNode("split_star", new SplitStringAt('*'));
        b.addNode("token_0", new TokenAt(0));
        b.addNode("token_1", new TokenAt(1));
        b.addNode("parse", new ParseNumber());
        b.addNode("num_to_mat", new NumberToMatrix());
        b.addNode("compose", new ComposeMatrices());
        b.addNode("mat_to_num", new MatrixToNumber());
        b.addNode("output", new OutputPrimitive());
        return b;
    }

    private static void printSideBySide(Map<String, Integer> a, Map<String, Integer> b, int aRuns, int bRuns) {
        TreeMap<String, double[]> rows = new TreeMap<>();
        a.forEach((k, v) -> rows.computeIfAbsent(canonicalize(k), kk -> new double[2])[0] = aRuns == 0 ? 0 : (double) v / aRuns);
        b.forEach((k, v) -> rows.computeIfAbsent(canonicalize(k), kk -> new double[2])[1] = bRuns == 0 ? 0 : (double) v / bRuns);
        rows.forEach((k, v) -> System.out.printf("  %-20s %-15.2f %-15.2f%n", k, v[0], v[1]));
    }

    /** Treat MlpPrimitive and TransformerPrimitive as the same logical role for comparison. */
    private static String canonicalize(String klass) {
        if (klass.equals("MlpPrimitive") || klass.equals("TransformerPrimitive")) {
            return "LearnedArith";
        }
        return klass;
    }

    private static boolean primCountsMatch(Stats sA, Stats sB) {
        Map<String, Double> a = avgByCanonical(sA.primCounts, sA.runs);
        Map<String, Double> b = avgByCanonical(sB.primCounts, sB.runs);
        if (!a.keySet().equals(b.keySet())) return false;
        for (var e : a.entrySet()) {
            if (Math.abs(e.getValue() - b.get(e.getKey())) > 0.5) return false;
        }
        return true;
    }

    private static Map<String, Double> avgByCanonical(Map<String, Integer> counts, int runs) {
        Map<String, Double> out = new TreeMap<>();
        counts.forEach((k, v) -> out.merge(canonicalize(k), runs == 0 ? 0.0 : (double) v / runs, Double::sum));
        return out;
    }

    private static final class Stats {
        int runs = 0;
        long totalNodes = 0;
        Map<String, Integer> primCounts = new TreeMap<>();
        int productOnPath = 0;

        void observe(CarvingResult r, double expectedProduct) {
            runs++;
            totalNodes += r.graph().nodes().size();
            Map<String, Integer> here = new HashMap<>();
            for (CompGraphNode n : r.graph().topoOrder()) {
                Primitive p = n.tNode().primitive();
                here.merge(p.getClass().getSimpleName(), 1, Integer::sum);
            }
            here.forEach((k, v) -> primCounts.merge(k, v, Integer::sum));
            if (Double.isNaN(expectedProduct)) return;
            for (CompGraphNode n : r.graph().topoOrder()) {
                if (n.producedValue() instanceof NumberValue(double nv)
                        && Math.abs(nv - expectedProduct) < 0.5
                        && r.graph().reaches(n, r.graph().terminal())) {
                    productOnPath++;
                    break;
                }
            }
        }
    }
}
