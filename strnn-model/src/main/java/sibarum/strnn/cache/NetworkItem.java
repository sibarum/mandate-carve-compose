package sibarum.strnn.cache;

import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.carving.RootBinding;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Objects;

/**
 * A cached entry that stores a trained subgraph and exposes it as a callable
 * function. The Key-Network counterpart to {@link EmbeddingItem}: instead of
 * returning a stored vector, the cache returns *behaviour* parameterized by
 * a query input.
 *
 * Construction takes a {@link CarvingResult} — the trained-state snapshot
 * the carver produced. The item retains a reference to the underlying
 * {@link ComputationGraph}; trainable state inside that graph (e.g. learned
 * bridges and embedding tables) lives where it is and continues to evolve
 * if the caller chooses to keep training. Most uses freeze the inner
 * carving at construction.
 *
 * The single declared input slot for {@link #execute} is the carving's
 * root binding. This is the simplest case — one input, one output — and
 * matches the shape every demo through v3 P5 has used. Multi-input cached
 * networks are a straightforward extension by carrying a list of slots
 * rather than a single one.
 */
public final class NetworkItem implements CachedItem {
    private final ComputationGraph graph;
    private final RootBinding inputBinding;
    private final ValueType inputType;
    private final ValueType outputType;

    public NetworkItem(
            ComputationGraph graph,
            RootBinding inputBinding,
            ValueType inputType,
            ValueType outputType) {
        this.graph = Objects.requireNonNull(graph);
        this.inputBinding = Objects.requireNonNull(inputBinding);
        this.inputType = Objects.requireNonNull(inputType);
        this.outputType = Objects.requireNonNull(outputType);
    }

    /**
     * Convenience factory: take a CarvingResult that has exactly one root
     * binding and treat it as a single-input network. Most carvings in the
     * KV-cache line have this shape.
     */
    public static NetworkItem fromCarving(
            CarvingResult carving, ValueType inputType, ValueType outputType) {
        if (carving.rootBindings().size() != 1) {
            throw new IllegalArgumentException(
                    "NetworkItem.fromCarving expects exactly one root binding; got "
                            + carving.rootBindings().size());
        }
        return new NetworkItem(
                carving.graph(),
                carving.rootBindings().getFirst(),
                inputType,
                outputType);
    }

    public ValueType inputType() {
        return inputType;
    }

    public ValueType outputType() {
        return outputType;
    }

    public ComputationGraph graph() {
        return graph;
    }

    /**
     * Rebind the inner graph's root slot to {@code input} and execute the
     * full graph. Returns the terminal's value. The graph's intermediate
     * node values are mutated as a side-effect of execution; callers that
     * want repeatability with a particular input must re-execute, not cache
     * intermediates.
     */
    public Value execute(Value input) {
        if (input.type() != inputType) {
            throw new IllegalArgumentException(
                    "NetworkItem expects input type " + inputType
                            + " but got " + input.type());
        }
        graph.bindRoot(inputBinding.node(), inputBinding.slot(), input);
        return graph.execute();
    }
}
