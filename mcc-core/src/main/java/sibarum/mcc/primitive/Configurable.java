package sibarum.mcc.primitive;

import java.util.Map;

/**
 * Optional contract for primitives with structural parameters (shape,
 * sizes, slice ranges, etc.) that must be serialized so the runtime
 * can reconstruct them. Stateless primitives need not implement this;
 * the exporter treats their config as empty.
 *
 * <p>Returned values must be JSON-serializable: numbers, strings,
 * booleans, lists thereof, or nested string-keyed maps.
 */
public interface Configurable {
    Map<String, Object> config();
}
