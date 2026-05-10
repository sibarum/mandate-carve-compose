package sibarum.strnn.demo;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.LearnedArithmetic;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.MlpPrimitive;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.primitive.TransformerPrimitive;
import sibarum.strnn.training.Datasets;
import sibarum.strnn.training.Example;
import sibarum.strnn.training.Pruner;
import sibarum.strnn.training.Trainer;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.transformer.Transformer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Pattern B: competitive coexistence. Both MLP and Transformer fill the
 * ADD and MUL learned-arithmetic slots in the same transformation graph.
 * The carver picks among them at random (initially) and increasingly by
 * edge-stats score (over training). After training, primitive-level
 * competition pruning checks whether the framework correctly identifies
 * the underperforming architecture.
 *
 * The setup is intentionally <i>asymmetric</i> at MUL: mlp_mul is
 * pretrained heavily, tfm_mul lightly. ADD is pretrained equally for
 * both. Expected diagnostic outcome:
 *   - tfm_mul's outgoing edges accumulate lower terminal scores.
 *   - prunePrimitiveCompetition removes tfm_mul (and only tfm_mul).
 *   - mlp_add and tfm_add survive as roughly comparable.
 *
 * If pruning instead removes the wrong primitive, or both ADD primitives,
 * we have a real signal that the framework's selection mechanism is not
 * working as advertised.
 */
public final class PatternBDemo {
    private static final int TRAIN_STEPS = 6000;
    private static final int LOG_EVERY = 1000;
    private static final double MLP_LR = 0.002;
    private static final double EXPLORATION_EPSILON = 0.10;

    public static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        Transformer addTfm = new Transformer(2, 1, 16, 64, 1, 11L);
        Transformer mulTfm = new Transformer(2, 1, 32, 128, 1, 7L);

        System.out.println("Pretraining (asymmetric: tfm_mul gets a fraction of the budget)...");
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainTransformerSilent(addTfm, MlpRole.ADD, 4000, 32, 0.005);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);
        TrainingDemoBootstrap.pretrainTransformerSilent(mulTfm, MlpRole.MUL, 2000, 32, 0.0015);

        System.out.println("\nPre-training accuracy snapshot (single-digit MAE):");
        System.out.printf("  mlp_add  MAE=%.3f%n", evalMlp(addMlp, true));
        System.out.printf("  tfm_add  MAE=%.3f%n", evalTransformer(addTfm, true));
        System.out.printf("  mlp_mul  MAE=%.3f%n", evalMlp(mulMlp, false));
        System.out.printf("  tfm_mul  MAE=%.3f  <-- intentionally undertrained%n", evalTransformer(mulTfm, false));

        TransformationGraph tg = buildTransformationGraph(addMlp, addTfm, mulMlp, mulTfm);
        Trainer trainer = new Trainer(tg, /*carverSeed=*/7L, MLP_LR,
                /*pruner=*/null, /*pruneEvery=*/0, EXPLORATION_EPSILON);  // no in-loop pruning
        System.out.printf("Carver exploration epsilon: %.2f%n", EXPLORATION_EPSILON);

        System.out.printf("%nTraining %d examples (no in-loop pruning)...%n", TRAIN_STEPS);
        Random rng = new Random(11L);
        double rollScore = 0;
        int rollCount = 0;
        int rollHits = 0;
        for (int i = 1; i <= TRAIN_STEPS; i++) {
            Example ex = nextExample(rng);
            Trainer.StepResult sr = trainer.step(ex);
            rollScore += sr.score();
            rollCount++;
            if (sr.allSatisfied()) rollHits++;
            if (i % LOG_EVERY == 0) {
                System.out.printf("step %5d  avg-score=%.3f  all-mandates=%.3f%n",
                        i, rollScore / rollCount, (double) rollHits / rollCount);
                rollScore = 0;
                rollCount = 0;
                rollHits = 0;
            }
        }

        System.out.println("\nPer-primitive aggregate edge stats:");
        printPerPrimitiveStats(tg);

        System.out.println("\nPer-edge stats for LearnedArithmetic primitives (sorted by samples desc):");
        printPerEdgeStatsForLearned(tg);

        Pruner pruner = new Pruner(/*minSamples=*/300, /*margin=*/0.03);
        int prunedFromCompetition = pruner.prunePrimitiveCompetition(tg);
        System.out.printf("%nprunePrimitiveCompetition: %d edges pruned%n", prunedFromCompetition);

        System.out.println("\nPost-prune state per primitive:");
        for (TransformationNode tn : tg.nodes()) {
            if (!(tn.primitive() instanceof LearnedArithmetic la)) continue;
            long pruned = countPrunedEdges(tg, tn);
            long total = tg.outgoing(tn).size() + tg.incoming(tn).size();
            String tag = pruned > 0 ? "  <-- PRUNED" : "";
            System.out.printf("  %-12s role=%s  pruned-edges=%d/%d%s%n",
                    tn.id(), la.role(), pruned, total, tag);
        }

        System.out.println("\nPattern B run complete.");
    }

    private static long countPrunedEdges(TransformationGraph tg, TransformationNode tn) {
        long count = 0;
        for (TransformationEdge e : tg.outgoing(tn)) if (e.stats().isPruned()) count++;
        for (TransformationEdge e : tg.incoming(tn)) if (e.stats().isPruned()) count++;
        return count;
    }

    private static void printPerPrimitiveStats(TransformationGraph tg) {
        Map<MlpRole, List<TransformationNode>> byRole = new EnumMap<>(MlpRole.class);
        for (TransformationNode tn : tg.nodes()) {
            if (tn.primitive() instanceof LearnedArithmetic la) {
                byRole.computeIfAbsent(la.role(), k -> new ArrayList<>()).add(tn);
            }
        }
        for (var entry : byRole.entrySet()) {
            System.out.printf("  role %s:%n", entry.getKey());
            List<TransformationNode> sorted = new ArrayList<>(entry.getValue());
            sorted.sort(Comparator.comparing(TransformationNode::id));
            for (TransformationNode tn : sorted) {
                double avgOut = avgEdgeScore(tg.outgoing(tn));
                double avgIn = avgEdgeScore(tg.incoming(tn));
                long samplesOut = totalSamples(tg.outgoing(tn));
                long samplesIn = totalSamples(tg.incoming(tn));
                System.out.printf("    %-12s  out=%.3f (n=%d)   in=%.3f (n=%d)%n",
                        tn.id(), avgOut, samplesOut, avgIn, samplesIn);
            }
        }
    }

    private static void printPerEdgeStatsForLearned(TransformationGraph tg) {
        List<TransformationEdge> learnedEdges = new ArrayList<>();
        for (TransformationEdge e : tg.edges()) {
            if (e.from().primitive() instanceof LearnedArithmetic
                    || e.to().primitive() instanceof LearnedArithmetic) {
                if (e.stats().samples() > 0) learnedEdges.add(e);
            }
        }
        learnedEdges.sort(Comparator.comparingLong((TransformationEdge e) -> e.stats().samples()).reversed());
        for (TransformationEdge e : learnedEdges) {
            System.out.printf("  %-15s -> %-15s  score=%.3f  n=%d%n",
                    e.from().id(), e.to().id(),
                    e.stats().score(), e.stats().samples());
        }
    }

    private static double avgEdgeScore(List<TransformationEdge> edges) {
        double sum = 0;
        int n = 0;
        for (TransformationEdge e : edges) {
            if (e.stats().samples() == 0) continue;
            sum += e.stats().score();
            n++;
        }
        return n == 0 ? Double.NaN : sum / n;
    }

    private static long totalSamples(List<TransformationEdge> edges) {
        long s = 0;
        for (TransformationEdge e : edges) s += e.stats().samples();
        return s;
    }

    private static Example nextExample(Random rng) {
        Datasets.Shape shape = rng.nextBoolean() ? Datasets.Shape.ADD_THEN_MUL : Datasets.Shape.MUL_THEN_ADD;
        return Datasets.generate(rng, shape);
    }

    private static double evalMlp(Mlp mlp, boolean isAdd) {
        double total = 0;
        int count = 0;
        double scale = NumberToMatrix.SCALE;
        for (int a = 0; a < 10; a++) for (int b = 0; b < 10; b++) {
            double[] in = {a / scale, b / scale};
            double pred = mlp.forward(in)[0] * scale;
            double truth = isAdd ? (a + b) : (a * b);
            total += Math.abs(pred - truth);
            count++;
        }
        return total / count;
    }

    private static double evalTransformer(Transformer t, boolean isAdd) {
        double total = 0;
        int count = 0;
        double scale = NumberToMatrix.SCALE;
        for (int a = 0; a < 10; a++) for (int b = 0; b < 10; b++) {
            double[] in = {a / scale, b / scale};
            double pred = t.forward(in)[0] * scale;
            double truth = isAdd ? (a + b) : (a * b);
            total += Math.abs(pred - truth);
            count++;
        }
        return total / count;
    }

    private static TransformationGraph buildTransformationGraph(Mlp addMlp, Transformer addTfm, Mlp mulMlp, Transformer mulTfm) {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("split_plus", new SplitStringAt('+'));
        b.addNode("split_star", new SplitStringAt('*'));
        b.addNode("token_0", new TokenAt(0));
        b.addNode("token_1", new TokenAt(1));
        b.addNode("parse", new ParseNumber());
        b.addNode("num_to_mat", new NumberToMatrix());
        b.addNode("compose", new ComposeMatrices());
        b.addNode("mlp_add", new MlpPrimitive(MlpRole.ADD, addMlp));
        b.addNode("tfm_add", new TransformerPrimitive(MlpRole.ADD, addTfm));
        b.addNode("mlp_mul", new MlpPrimitive(MlpRole.MUL, mulMlp));
        b.addNode("tfm_mul", new TransformerPrimitive(MlpRole.MUL, mulTfm));
        b.addNode("mat_to_num", new MatrixToNumber());
        b.addNode("output", new OutputPrimitive());
        return b.build();
    }
}
