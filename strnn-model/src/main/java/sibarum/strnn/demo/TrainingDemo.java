package sibarum.strnn.demo;

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
import sibarum.strnn.training.Pruner;
import sibarum.strnn.training.Trainer;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;

import java.util.Random;

/**
 * Phase 5: end-to-end training. Carves graphs, trains MLPs through mandate
 * supervision, accumulates edge stats, prunes when one edge consistently
 * lags another sharing its (srcType, dstType) pair.
 *
 * Reports periodic accuracy + mandate-satisfaction rates; final held-out
 * evaluation against the §9.5 step-2 task (a±b·c with single-digit operands).
 */
public final class TrainingDemo {
    private static final int TRAIN_STEPS = 8000;
    private static final int LOG_EVERY = 500;
    private static final int HELDOUT_SIZE = 200;
    private static final double MLP_LR = 0.003;
    private static final long PRUNE_EVERY = 2000;
    private static final long PRUNE_MIN_SAMPLES = 500;

    static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);

        System.out.println("Pretraining MLPs offline (curriculum, §6.7)...");
        TrainingDemoBootstrap.pretrain(addMlp, MlpRole.ADD, /*epochs=*/4000, /*batch=*/32, /*lr=*/0.01);
        TrainingDemoBootstrap.pretrain(mulMlp, MlpRole.MUL, /*epochs=*/12000, /*batch=*/32, /*lr=*/0.004);

        TransformationGraph tg = buildTransformationGraph(addMlp, mulMlp);
        Pruner pruner = new Pruner(PRUNE_MIN_SAMPLES, /*margin=*/0.15);
        Trainer trainer = new Trainer(tg, /*carverSeed=*/7L, MLP_LR, pruner, PRUNE_EVERY);

        Random rng = new Random(11L);
        double rollingScore = 0.0;
        int rollingCount = 0;
        int rollingHits = 0;
        int rollingCarveFails = 0;

        System.out.println("Training " + TRAIN_STEPS + " examples...");
        for (int i = 1; i <= TRAIN_STEPS; i++) {
            Example ex = nextExample(rng);
            Trainer.StepResult sr = trainer.step(ex);
            rollingCount++;
            rollingScore += sr.score();
            if (sr.allSatisfied()) rollingHits++;
            if (sr.error() != null && sr.error().equals("carve_failed")) rollingCarveFails++;
            if (sr.prunedThisStep() > 0) {
                System.out.printf("  step %d: pruned %d edges (total pruned %d)%n",
                        i, sr.prunedThisStep(), trainer.totalPruned());
            }

            if (i % LOG_EVERY == 0) {
                double avgScore = rollingScore / rollingCount;
                double hitRate = (double) rollingHits / rollingCount;
                double failRate = (double) rollingCarveFails / rollingCount;
                System.out.printf(
                        "step %5d  avg-score=%.3f  all-mandates=%.3f  carve-fails=%.3f%n",
                        i, avgScore, hitRate, failRate);
                rollingScore = 0.0;
                rollingCount = 0;
                rollingHits = 0;
                rollingCarveFails = 0;
            }
        }

        System.out.println("\nHeld-out evaluation (" + HELDOUT_SIZE + " unseen examples):");
        Random heldRng = new Random(2026L);
        int correct = 0;
        int allMandates = 0;
        for (int i = 0; i < HELDOUT_SIZE; i++) {
            Example ex = nextExample(heldRng);
            Trainer.StepResult sr = trainer.step(ex);
            if (sr.allSatisfied()) allMandates++;
            sibarum.strnn.mandate.Mandate resultMandate = ex.mandates().result();
            if (resultMandate != null
                    && sr.executedResult() instanceof sibarum.strnn.value.NumberValue(double n)
                    && resultMandate.expected() instanceof sibarum.strnn.value.NumberValue(double expected)
                    && Math.abs(n - expected) <= 0.5) {
                correct++;
            }
        }
        System.out.printf("held-out accuracy (|err|<=0.5): %.3f%n", (double) correct / HELDOUT_SIZE);
        System.out.printf("held-out all-mandates rate:    %.3f%n", (double) allMandates / HELDOUT_SIZE);

        long pruned = tg.edges().stream().filter(e -> e.stats().isPruned()).count();
        System.out.printf("%npruned edges: %d / %d  (trainer reports total pruned = %d)%n",
                pruned, tg.edges().size(), trainer.totalPruned());

        System.out.println("\nedge stats summary (sorted by samples desc):");
        tg.edges().stream()
                .sorted((a, b) -> Long.compare(b.stats().samples(), a.stats().samples()))
                .forEach(e -> System.out.printf("  %-30s -> %-15s  score=%.3f  n=%d%s%n",
                        e.from().id(), e.to().id(),
                        e.stats().score(), e.stats().samples(),
                        e.stats().isPruned() ? "  PRUNED" : ""));

        System.out.println("\nPhase 5 training run complete.");
    }

    private static Example nextExample(Random rng) {
        Datasets.Shape shape = rng.nextBoolean() ? Datasets.Shape.ADD_THEN_MUL : Datasets.Shape.MUL_THEN_ADD;
        return Datasets.generate(rng, shape);
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
}
