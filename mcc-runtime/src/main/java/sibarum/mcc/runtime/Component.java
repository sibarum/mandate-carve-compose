package sibarum.mcc.runtime;

import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.primitive.BuiltinPrimitives;
import sibarum.mcc.primitive.PrimitiveRegistry;
import sibarum.mcc.serialization.Importer;
import sibarum.mcc.value.Value;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * The inference-time view of an exported {@code component.mcc}.
 *
 * <pre>
 *   Component c = Component.load(Path.of("model.mcc"));
 *   Value out = c.infer(Map.of("x", new MatrixValue(...)));
 * </pre>
 *
 * <p>{@code Component} is the only public API of {@code mcc-runtime} —
 * everything else is implementation detail. Loading routes through
 * {@link Importer} with the {@link BuiltinPrimitives#defaults} registry;
 * applications with custom primitives should construct an {@link Importer}
 * directly.
 *
 * <p>Inference: bind each named root input to the corresponding Value,
 * execute the graph, return the terminal node's produced Value.
 */
public final class Component {

    private final Importer.LoadedGraph loaded;

    private Component(Importer.LoadedGraph loaded) {
        this.loaded = loaded;
    }

    public static Component load(Path componentDir) throws IOException {
        return load(componentDir, BuiltinPrimitives.defaults());
    }

    public static Component load(Path componentDir, PrimitiveRegistry registry) throws IOException {
        Importer importer = new Importer(registry);
        return new Component(importer.load(componentDir));
    }

    public ComputationGraph graph() {
        return loaded.graph();
    }

    public Map<String, Importer.RootBinding> rootInputs() {
        return loaded.rootInputs();
    }

    /**
     * Bind named inputs to values, execute the graph, and return the
     * terminal value.
     *
     * @throws IllegalArgumentException if a required root input is
     *         missing from {@code inputs}
     */
    public Value infer(Map<String, Value> inputs) {
        for (Map.Entry<String, Importer.RootBinding> e : loaded.rootInputs().entrySet()) {
            Value v = inputs.get(e.getKey());
            if (v == null) {
                throw new IllegalArgumentException("missing required input: " + e.getKey());
            }
            Importer.RootBinding rb = e.getValue();
            loaded.graph().bindRoot(rb.node(), rb.slot(), v);
        }
        return loaded.graph().execute();
    }
}
