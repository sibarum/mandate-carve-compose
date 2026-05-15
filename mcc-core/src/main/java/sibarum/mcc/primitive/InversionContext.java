package sibarum.mcc.primitive;

import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * Read-only context handed to {@link Inversion#invert} so primitives can
 * make context-aware inversion choices without seeing the carver's
 * internals.
 *
 * <p>Surfaces (a) the root input that started the carving, (b) values
 * reachable forward from the root through other primitives in the
 * substrate, (c) the mandate value pool (intermediate / result values
 * the user supplied), and (d) an RNG for primitives that need to pick
 * among ambiguous inversions.
 */
public interface InversionContext {

    /** The root input value the carver is solving for. */
    Value rootInput();

    /**
     * Values of {@code type} that can be reached by forward-applying
     * non-Terminal primitives starting at the root input. Empty if no
     * such forward chain exists.
     */
    List<Value> reachableValuesOfType(ValueType type);

    /** Concrete values mentioned in the active mandate set. */
    List<Value> mandateValues();

    /**
     * A representative {@link MatrixValue} of the requested dim drawn
     * from the reachable / mandate / root pool. Used by trainable
     * primitives (e.g. {@link sibarum.mcc.op.Linear}) to pick a
     * concrete input anchor without committing to a specific upstream.
     */
    Optional<MatrixValue> anchorByMatrixDim(int dim);

    /** Shared RNG; primitives must use this rather than constructing their own. */
    Random rng();
}
