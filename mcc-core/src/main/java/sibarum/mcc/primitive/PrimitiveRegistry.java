package sibarum.mcc.primitive;

import java.util.HashMap;
import java.util.Map;

/**
 * Maps {@link Primitive#name()} (or a config-derived schema name) to
 * factory functions that re-instantiate a primitive from a config
 * dictionary. Used by the importer to reconstruct a graph from its
 * JSON descriptor.
 *
 * <p>Stateless primitives can register with empty config-free
 * factories; parameterized primitives need the config map to specify
 * their structural parameters (sizes, seeds, etc.).
 *
 * <p>Built-in primitives are registered via {@link BuiltinPrimitives}.
 */
public final class PrimitiveRegistry {

    private final Map<String, Factory> factories = new HashMap<>();

    /** Factory function: given a config map, produce a primitive. */
    public interface Factory {
        Primitive create(Map<String, Object> config);
    }

    public PrimitiveRegistry register(String name, Factory factory) {
        if (factories.put(name, factory) != null) {
            throw new IllegalArgumentException("primitive '" + name + "' already registered");
        }
        return this;
    }

    public Primitive create(String name, Map<String, Object> config) {
        Factory f = factories.get(name);
        if (f == null) {
            throw new IllegalArgumentException("no primitive registered for '" + name + "'");
        }
        return f.create(config == null ? Map.of() : config);
    }

    public boolean isRegistered(String name) {
        return factories.containsKey(name);
    }
}
