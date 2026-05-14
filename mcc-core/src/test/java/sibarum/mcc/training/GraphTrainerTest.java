package sibarum.mcc.training;

import org.junit.jupiter.api.Test;
import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.block.MlpBlock;
import sibarum.mcc.value.MatrixValue;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Trains an MLP via {@link GraphTrainer} on a synthetic regression
 * task and asserts that loss decreases substantially. This is the
 * Phase-4 verification — minimal but real: build a graph, train,
 * check convergence.
 */
class GraphTrainerTest {

    @Test
    void mlpConvergesOnSyntheticRegression() {
        // Target: y = [x[0] * x[1], x[0] + x[1]] on x ∈ [-1, 1]² (smooth bivariate).
        MlpBlock mlp = new MlpBlock(new int[] { 2, 16, 16, 2 }, 7L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("mlp", mlp)
                .build();
        CompGraphNode node = new CompGraphNode("c-mlp", tg.node("mlp"));
        ComputationGraph cg = new ComputationGraph(List.of(node), node);

        Corpus corpus = new SyntheticCorpus(256, 0L);

        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> {
                    MatrixValue x = (MatrixValue) ex.inputs().get("x");
                    g.bindRoot(g.nodes().get(0), 0, x);
                },
                0.05,
                /* stepEveryExample */ true
        );

        double loss0 = trainer.trainEpoch(corpus);
        double lossLast = loss0;
        for (int i = 0; i < 50; i++) {
            lossLast = trainer.trainEpoch(corpus);
        }

        assertTrue(Double.isFinite(lossLast), "final loss must be finite: " + lossLast);
        assertTrue(lossLast < loss0 * 0.1,
                "expected loss reduction by >10× after 50 epochs; loss0=" + loss0 + ", final=" + lossLast);
    }

    @Test
    void uniqueTrainablesDedupes() {
        MlpBlock mlp = new MlpBlock(new int[] { 2, 4, 1 }, 1L);
        // Two TransformationNodes wrapping the SAME MlpBlock instance — they share weights.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("mlp-a", mlp)
                .addNode("mlp-b", mlp)
                .build();
        CompGraphNode a = new CompGraphNode("c-a", tg.node("mlp-a"));
        CompGraphNode b = new CompGraphNode("c-b", tg.node("mlp-b"));
        ComputationGraph cg = new ComputationGraph(List.of(a, b), b);

        GraphTrainer t = new GraphTrainer(cg, (g, ex) -> {}, 0.01, false);
        assertEquals(1, t.uniqueTrainables().size(),
                "Two nodes wrapping the same Trainable must dedup to one unique trainable");
    }

    /** y = [x0·x1, x0+x1] on uniform x ∈ [-1, 1]². */
    private static final class SyntheticCorpus implements Corpus {
        private final long size;
        private final long seed;

        SyntheticCorpus(long size, long seed) {
            this.size = size;
            this.seed = seed;
        }

        @Override public String name() { return "synthetic-2x2"; }
        @Override public long size() { return size; }

        @Override
        public Iterator<Example> stream() {
            Random rng = new Random(seed);
            List<Example> all = new ArrayList<>((int) size);
            for (long i = 0; i < size; i++) {
                double x0 = rng.nextDouble() * 2 - 1;
                double x1 = rng.nextDouble() * 2 - 1;
                MatrixValue x = new MatrixValue(new double[] { x0, x1 });
                MatrixValue y = new MatrixValue(new double[] { x0 * x1, x0 + x1 });
                all.add(new Example("ex-" + i, Map.of("x", x), y));
            }
            return all.iterator();
        }
    }
}
