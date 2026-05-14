package sibarum.mcc.primitive;

import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Computational unit for the mcc-core graph substrate. Each primitive
 * declares a typed signature ({@link #inputTypes} → {@link #outputType})
 * and an {@link #apply} that consumes input values in order and
 * produces a single output value.
 *
 * <p>Primitives are stateless across calls unless they extend
 * {@link Trainable}, in which case implementations may cache
 * intermediate state from the most recent {@code apply} for use by a
 * subsequent {@code backward}.
 */
public interface Primitive {
    String name();

    List<ValueType> inputTypes();

    ValueType outputType();

    Value apply(List<Value> inputs);
}
