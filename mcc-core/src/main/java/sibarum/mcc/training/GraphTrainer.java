package sibarum.mcc.training;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.Value;

import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiConsumer;

/**
 * Minimal trainer for a {@link ComputationGraph} whose terminal node
 * wraps a {@link Trainable}. Each example:
 *
 * <ol>
 *   <li>The caller-provided {@code rootBinder} converts the example's
 *       named inputs into root bindings on the graph.</li>
 *   <li>{@link ComputationGraph#execute} runs forward.</li>
 *   <li>For every node whose primitive is {@link Trainable},
 *       {@code backward(example.target())} is invoked. Multiple
 *       trainables sharing the same {@link Trainable#trainableIdentity}
 *       receive backward only once.</li>
 *   <li>After the epoch (or after each example, depending on
 *       {@code stepEveryExample}), every unique trainable's
 *       {@code step(lr)} is invoked.</li>
 * </ol>
 *
 * <p>MVP scope: the target-based backward contract from {@link Trainable}
 * means this trainer only fully supports graphs whose Trainables can
 * accept the example's terminal target. For graphs with multiple
 * chained Trainables the per-trainable targeting is up to the caller
 * (e.g. by carving simulated values, a future capability).
 */
public final class GraphTrainer {

    private final ComputationGraph graph;
    private final BiConsumer<ComputationGraph, Example> rootBinder;
    private final double learningRate;
    private final boolean stepEveryExample;

    public GraphTrainer(ComputationGraph graph,
                        BiConsumer<ComputationGraph, Example> rootBinder,
                        double learningRate,
                        boolean stepEveryExample) {
        this.graph = graph;
        this.rootBinder = rootBinder;
        this.learningRate = learningRate;
        this.stepEveryExample = stepEveryExample;
    }

    public ComputationGraph graph() {
        return graph;
    }

    /**
     * Run one epoch through the corpus. Returns the running-mean
     * squared error against {@code Example.target} (for diagnostics).
     */
    public double trainEpoch(Corpus corpus) {
        IdentityHashMap<Object, Trainable> pendingSteps = new IdentityHashMap<>();
        double sumLoss = 0.0;
        long count = 0;
        Iterator<Example> it = corpus.stream();
        while (it.hasNext()) {
            Example ex = it.next();
            rootBinder.accept(graph, ex);
            Value out = graph.execute();
            sumLoss += halfMseToTarget(out, ex.target());
            backwardAllTrainables(ex.target(), pendingSteps);
            count++;
            if (stepEveryExample) {
                stepPending(pendingSteps);
            }
        }
        if (!stepEveryExample) {
            stepPending(pendingSteps);
        }
        return count == 0 ? 0.0 : sumLoss / count;
    }

    private void backwardAllTrainables(Value target, IdentityHashMap<Object, Trainable> pending) {
        IdentityHashMap<Object, Boolean> doneThisExample = new IdentityHashMap<>();
        for (CompGraphNode n : graph.nodes()) {
            if (n.tNode().primitive() instanceof Trainable t) {
                Object id = t.trainableIdentity();
                if (doneThisExample.containsKey(id)) continue;
                doneThisExample.put(id, Boolean.TRUE);
                t.backward(target);
                pending.put(id, t);
            }
        }
    }

    private void stepPending(IdentityHashMap<Object, Trainable> pending) {
        for (Trainable t : pending.values()) {
            t.step(learningRate);
        }
        pending.clear();
    }

    private static double halfMseToTarget(Value out, Value target) {
        // Only computes a meaningful loss for MatrixValue targets in MVP;
        // returns NaN for other types so logging can ignore them. Mandate-based
        // verification is the proper success metric.
        if (out instanceof sibarum.mcc.value.MatrixValue mo
                && target instanceof sibarum.mcc.value.MatrixValue mt
                && mo.data().length == mt.data().length) {
            double s = 0.0;
            for (int i = 0; i < mo.data().length; i++) {
                double d = mo.data()[i] - mt.data()[i];
                s += d * d;
            }
            return 0.5 * s;
        }
        return Double.NaN;
    }

    /**
     * Materialize a list of all unique Trainables in the graph (deduped
     * by {@link Trainable#trainableIdentity}). Useful for exporters
     * iterating parameters.
     */
    public List<Trainable> uniqueTrainables() {
        IdentityHashMap<Object, Trainable> seen = new IdentityHashMap<>();
        for (CompGraphNode n : graph.nodes()) {
            if (n.tNode().primitive() instanceof Trainable t) {
                seen.putIfAbsent(t.trainableIdentity(), t);
            }
        }
        return List.copyOf(seen.values());
    }
}
