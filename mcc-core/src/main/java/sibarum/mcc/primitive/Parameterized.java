package sibarum.mcc.primitive;

import java.util.List;
import java.util.Map;

/**
 * Optional contract for primitives whose internal parameters can be
 * serialized and restored. Implementations enumerate their parameters
 * as named tensors (shape + flat data) and accept a same-shaped map
 * to overwrite their state.
 *
 * <p>Parameter names live in the primitive's local namespace
 * (e.g. {@code "W"}, {@code "b"}). The {@code Exporter} qualifies
 * them with the owning node id when writing the export.
 */
public interface Parameterized {

    /**
     * Named tensor descriptor. {@code shape} is the logical shape;
     * {@code data} is row-major contiguous in a {@code double[]}.
     * Implementations should return defensive copies to keep the
     * snapshot decoupled from live state.
     */
    record NamedTensor(String name, int[] shape, double[] data) {}

    /** Snapshot of every learnable tensor owned by this primitive. */
    List<NamedTensor> parameters();

    /**
     * Restore parameters from a map keyed by the {@link NamedTensor#name}
     * returned by {@link #parameters()}. Shape must match; implementations
     * should throw {@link IllegalArgumentException} on mismatch.
     */
    void loadParameters(Map<String, NamedTensor> tensors);
}
