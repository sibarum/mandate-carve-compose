package sibarum.mcc.training;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * Trainer for a {@link ComputationGraph}. Per example:
 *
 * <ol>
 *   <li>{@code rootBinder} binds the example's named inputs to graph roots.</li>
 *   <li>{@code graph.execute()} runs forward.</li>
 *   <li>The terminal gradient is computed as {@code output − target}
 *       (MSE convention). It flows backward through the graph's
 *       reverse topological order; at each {@link Differentiable} node,
 *       {@code backward(gradAtOutput)} produces input-slot gradients
 *       that are accumulated into the upstream nodes' incoming-gradient
 *       slots. Every {@link Trainable}'s {@code step(lr)} is invoked
 *       once per epoch (or once per example, see
 *       {@code stepEveryExample}), identity-deduped via
 *       {@link Trainable#trainableIdentity}.</li>
 * </ol>
 *
 * <p>Multi-trainable chains (Embed → Linear → Relu → Linear → Softmax,
 * etc.) train through the standard gradient flow.
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
     * Run one epoch through the corpus. Returns mean half-MSE loss
     * for diagnostics.
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
            sumLoss += halfMse(out, ex.target());
            Value terminalGrad = mseGrad(out, ex.target());
            backpropFromTerminal(terminalGrad, pendingSteps);
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

    private void backpropFromTerminal(Value terminalGrad, IdentityHashMap<Object, Trainable> pendingSteps) {
        Map<CompGraphNode, Value> grads = new LinkedHashMap<>();
        grads.put(graph.terminal(), terminalGrad);

        List<CompGraphNode> order = graph.topoOrder();
        for (int i = order.size() - 1; i >= 0; i--) {
            CompGraphNode n = order.get(i);
            Value gradAtN = grads.get(n);
            if (gradAtN == null) continue;

            Primitive p = n.tNode().primitive();
            if (!(p instanceof Differentiable diff)) continue;

            List<Value> inputGrads = diff.backward(gradAtN);
            if (p instanceof Trainable t) {
                pendingSteps.put(t.trainableIdentity(), t);
            }

            for (int slot = 0; slot < n.slotCount(); slot++) {
                SlotSource src = n.slot(slot);
                if (src == null) continue;
                if (slot >= inputGrads.size()) break;
                Value g = inputGrads.get(slot);
                if (g == null) continue;
                accumulate(grads, src.source(), g);
            }
        }
    }

    private static void accumulate(Map<CompGraphNode, Value> grads, CompGraphNode key, Value addend) {
        Value existing = grads.get(key);
        if (existing == null) {
            grads.put(key, addend);
        } else {
            grads.put(key, addValues(existing, addend));
        }
    }

    private static Value addValues(Value a, Value b) {
        if (a instanceof MatrixValue ma && b instanceof MatrixValue mb) {
            if (ma.data().length != mb.data().length) {
                throw new IllegalStateException(
                        "gradient accumulator dim mismatch: " + ma.data().length + " vs " + mb.data().length);
            }
            double[] out = new double[ma.data().length];
            for (int i = 0; i < out.length; i++) out[i] = ma.data()[i] + mb.data()[i];
            return new MatrixValue(out);
        }
        if (a instanceof NumberValue na && b instanceof NumberValue nb) {
            return new NumberValue(na.n() + nb.n());
        }
        throw new IllegalStateException(
                "cannot accumulate gradients of type " + a.type() + " and " + b.type());
    }

    private void stepPending(IdentityHashMap<Object, Trainable> pending) {
        for (Trainable t : pending.values()) {
            t.step(learningRate);
        }
        pending.clear();
    }

    private static double halfMse(Value out, Value target) {
        if (out instanceof MatrixValue mo && target instanceof MatrixValue mt
                && mo.data().length == mt.data().length) {
            double s = 0.0;
            for (int i = 0; i < mo.data().length; i++) {
                double d = mo.data()[i] - mt.data()[i];
                s += d * d;
            }
            return 0.5 * s;
        }
        if (out instanceof NumberValue no && target instanceof NumberValue nt) {
            double d = no.n() - nt.n();
            return 0.5 * d * d;
        }
        return Double.NaN;
    }

    private static Value mseGrad(Value out, Value target) {
        if (out instanceof MatrixValue mo && target instanceof MatrixValue mt) {
            double[] g = new double[mo.data().length];
            for (int i = 0; i < g.length; i++) g[i] = mo.data()[i] - mt.data()[i];
            return new MatrixValue(g);
        }
        if (out instanceof NumberValue no && target instanceof NumberValue nt) {
            return new NumberValue(no.n() - nt.n());
        }
        throw new IllegalArgumentException(
                "GraphTrainer: cannot produce MSE gradient for terminal type " + out.type());
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
        return Collections.unmodifiableList(List.copyOf(seen.values()));
    }
}
