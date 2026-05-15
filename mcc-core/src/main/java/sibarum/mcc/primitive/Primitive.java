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
 * {@link Trainable} or {@link Differentiable}, in which case
 * implementations may cache intermediate state from the most recent
 * {@code apply} for use by a subsequent {@code backward}.
 *
 * <p>{@link #inversion} returns the primitive's inversion strategy
 * for the carver. Stateless deterministic primitives can usually
 * invert exactly; trainable primitives typically return one of
 * several plausible candidates via {@link InversionContext}. The
 * default is {@link Inversion#NONE} — primitives that don't override
 * it can still appear in graphs but are dead ends for the carver.
 */
public interface Primitive {
    String name();

    List<ValueType> inputTypes();

    ValueType outputType();

    Value apply(List<Value> inputs);

    default Inversion inversion() {
        return Inversion.NONE;
    }
}
