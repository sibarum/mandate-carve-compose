package sibarum.mcc.training;

import org.junit.jupiter.api.Test;
import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.Relu;
import sibarum.mcc.value.MatrixValue;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The decisive test for the autograd seam introduced in (A): a graph
 * with multiple chained {@link sibarum.mcc.primitive.Trainable}s —
 * Linear → Relu → Linear → terminal — must train end-to-end.
 *
 * <p>Before (A), the old target-based {@code Trainable.backward} meant
 * only one node could carry the loss signal at a time; this test
 * would not converge (the upstream Linear's gradient was decoupled
 * from the downstream loss). With the new gradient-flow trainer, both
 * Linears train against the terminal MSE loss, with the gradient
 * flowing through Relu in between.
 */
class MultiTrainableChainTest {

    @Test
    void linearReluLinearConvergesOnSyntheticRegression() {
        // Two trainable Linears in series with a Relu between.
        Linear l1 = new Linear(8, 2, true, 1L);
        Relu relu = new Relu();
        Linear l2 = new Linear(2, 8, true, 2L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("l1", l1)
                .addNode("relu", relu)
                .addNode("l2", l2)
                .build();

        CompGraphNode n1 = new CompGraphNode("c-l1", tg.node("l1"));
        CompGraphNode n2 = new CompGraphNode("c-relu", tg.node("relu"));
        CompGraphNode n3 = new CompGraphNode("c-l2", tg.node("l2"));
        n2.wire(0, new SlotSource(n1, tg.edge(tg.node("l1"), tg.node("relu"))));
        n3.wire(0, new SlotSource(n2, tg.edge(tg.node("relu"), tg.node("l2"))));
        ComputationGraph cg = new ComputationGraph(List.of(n1, n2, n3), n3);

        Corpus corpus = new SyntheticCorpus(256, 0L);

        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> {
                    MatrixValue x = (MatrixValue) ex.inputs().get("x");
                    g.bindRoot(n1, 0, x);
                },
                0.05,
                /* stepEveryExample */ true
        );

        double loss0 = trainer.trainEpoch(corpus);
        double lossLast = loss0;
        for (int i = 0; i < 100; i++) lossLast = trainer.trainEpoch(corpus);

        assertTrue(Double.isFinite(lossLast), "final loss must be finite: " + lossLast);
        assertTrue(lossLast < loss0 * 0.2,
                "expected >5× loss reduction in chained Linear→Relu→Linear; "
                        + "loss0=" + loss0 + ", final=" + lossLast);
    }

    @Test
    void gradientReachesUpstreamTrainable() {
        // Sanity: after one backward, the upstream Linear must have non-zero
        // accumulated gradients in its pendingDw — proving the gradient
        // flowed through Relu and into l1.
        Linear l1 = new Linear(4, 2, true, 1L);
        Relu relu = new Relu();
        Linear l2 = new Linear(2, 4, true, 2L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("l1", l1)
                .addNode("relu", relu)
                .addNode("l2", l2)
                .build();

        CompGraphNode n1 = new CompGraphNode("c-l1", tg.node("l1"));
        CompGraphNode n2 = new CompGraphNode("c-relu", tg.node("relu"));
        CompGraphNode n3 = new CompGraphNode("c-l2", tg.node("l2"));
        n2.wire(0, new SlotSource(n1, tg.edge(tg.node("l1"), tg.node("relu"))));
        n3.wire(0, new SlotSource(n2, tg.edge(tg.node("relu"), tg.node("l2"))));
        ComputationGraph cg = new ComputationGraph(List.of(n1, n2, n3), n3);

        // Snapshot upstream weights.
        double w1Before = l1.weights()[0][0];

        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> g.bindRoot(n1, 0, (MatrixValue) ex.inputs().get("x")),
                0.1,
                true
        );
        trainer.trainEpoch(new TinyCorpus());

        double w1After = l1.weights()[0][0];
        assertNotEquals(w1Before, w1After,
                "upstream Linear's weights did not change — gradient never reached it");
    }

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
                // Non-linear target: y = [x0², x0*x1] (Relu chain helps here).
                MatrixValue y = new MatrixValue(new double[] { x0 * x0, x0 * x1 });
                all.add(new Example("ex-" + i, Map.of("x", x), y));
            }
            return all.iterator();
        }
    }

    private static final class TinyCorpus implements Corpus {
        @Override public String name() { return "tiny"; }
        @Override public long size() { return 4; }
        @Override
        public Iterator<Example> stream() {
            // Four examples; outputs deliberately non-trivial so gradients
            // are non-zero with high probability.
            return List.of(
                    new Example("a", Map.of("x", new MatrixValue(new double[] {  0.5,  0.5 })),
                            new MatrixValue(new double[] { 1, 2, 3, 4 })),
                    new Example("b", Map.of("x", new MatrixValue(new double[] { -0.5,  0.5 })),
                            new MatrixValue(new double[] { 4, 3, 2, 1 })),
                    new Example("c", Map.of("x", new MatrixValue(new double[] {  0.5, -0.5 })),
                            new MatrixValue(new double[] { 1, 0, 1, 0 })),
                    new Example("d", Map.of("x", new MatrixValue(new double[] { -0.5, -0.5 })),
                            new MatrixValue(new double[] { 0, 1, 0, 1 }))
            ).iterator();
        }
    }
}
