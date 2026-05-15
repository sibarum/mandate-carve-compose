package sibarum.mcc.training;

import org.junit.jupiter.api.Test;
import sibarum.mcc.graph.substrate.TransformationEdge;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.graph.substrate.TransformationNode;
import sibarum.mcc.op.Add;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the {@link Pruner} edge-pair-competition loop end-to-end:
 *
 * <ol>
 *   <li>Build a substrate where two source nodes share the same
 *       (output, dst-input) pair and both feed a third node.</li>
 *   <li>Simulate training updates: one source's outgoing edge accumulates
 *       high scores; the other accumulates low scores.</li>
 *   <li>Run {@link Pruner#prune}: the lower-scoring edge gets pruned,
 *       the higher-scoring edge survives.</li>
 * </ol>
 */
class PrunerTest {

    @Test
    void prunesLowerScoringEdge() {
        // Three nodes: src1 and src2 both produce MATRIX; dst takes (MATRIX, MATRIX).
        // Substrate builder gives every (output, dst-input) pair an edge automatically.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("src1", new Add())  // (MATRIX, MATRIX) -> MATRIX
                .addNode("src2", new Add())
                .addNode("dst",  new Add())
                .build();

        TransformationNode src1 = tg.node("src1");
        TransformationNode src2 = tg.node("src2");
        TransformationNode dst  = tg.node("dst");

        TransformationEdge eGood = tg.edge(src1, dst);
        TransformationEdge eBad  = tg.edge(src2, dst);
        assertNotNull(eGood);
        assertNotNull(eBad);

        // Simulate evaluator updates: src1's edge sees a stream of high
        // rewards, src2's a stream of low rewards.
        for (int i = 0; i < 50; i++) eGood.stats().update(0.95);
        for (int i = 0; i < 50; i++) eBad.stats().update(0.05);

        // Both edges share the (MATRIX → MATRIX) pair so they compete.
        Pruner pruner = new Pruner(/* minSamples */ 10, /* margin */ 0.3);
        int pruned = pruner.prune(tg);

        assertTrue(pruned >= 1, "expected at least one edge to be pruned, got " + pruned);
        assertFalse(eGood.stats().isPruned(), "high-scoring edge must NOT be pruned");
        assertTrue(eBad.stats().isPruned(), "low-scoring edge must be pruned");
    }

    @Test
    void doesNotPruneWhenSamplesBelowThreshold() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("src1", new Add())
                .addNode("src2", new Add())
                .addNode("dst",  new Add())
                .build();
        TransformationEdge eGood = tg.edge(tg.node("src1"), tg.node("dst"));
        TransformationEdge eBad  = tg.edge(tg.node("src2"), tg.node("dst"));

        // Score gap is big but only 3 samples each (below minSamples = 10).
        for (int i = 0; i < 3; i++) eGood.stats().update(0.95);
        for (int i = 0; i < 3; i++) eBad.stats().update(0.05);

        Pruner pruner = new Pruner(10, 0.3);
        int pruned = pruner.prune(tg);
        assertEquals(0, pruned, "must not prune without enough samples to trust the score");
        assertFalse(eBad.stats().isPruned());
    }

    @Test
    void doesNotPruneWithinMargin() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("src1", new Add())
                .addNode("src2", new Add())
                .addNode("dst",  new Add())
                .build();
        TransformationEdge eA = tg.edge(tg.node("src1"), tg.node("dst"));
        TransformationEdge eB = tg.edge(tg.node("src2"), tg.node("dst"));

        // Both edges score similarly — the gap is below margin (0.3).
        for (int i = 0; i < 50; i++) eA.stats().update(0.60);
        for (int i = 0; i < 50; i++) eB.stats().update(0.55);

        int pruned = new Pruner(10, 0.3).prune(tg);
        assertEquals(0, pruned, "edges within margin should not be pruned");
        assertFalse(eA.stats().isPruned());
        assertFalse(eB.stats().isPruned());
    }

    @Test
    void prunedEdgeStaysPrunedAcrossMultipleRuns() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("src1", new Add())
                .addNode("src2", new Add())
                .addNode("dst",  new Add())
                .build();
        TransformationEdge eGood = tg.edge(tg.node("src1"), tg.node("dst"));
        TransformationEdge eBad  = tg.edge(tg.node("src2"), tg.node("dst"));

        for (int i = 0; i < 50; i++) eGood.stats().update(0.95);
        for (int i = 0; i < 50; i++) eBad.stats().update(0.05);

        Pruner pruner = new Pruner(10, 0.3);
        pruner.prune(tg);
        assertTrue(eBad.stats().isPruned());

        // Second invocation must not re-process the already-pruned edge.
        int pruned = pruner.prune(tg);
        assertEquals(0, pruned, "second run must not double-count an already-pruned edge");
        assertTrue(eBad.stats().isPruned(), "edge stays pruned");
    }
}
