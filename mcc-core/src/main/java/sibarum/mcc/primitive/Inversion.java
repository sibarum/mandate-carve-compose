package sibarum.mcc.primitive;

import sibarum.mcc.value.Value;

import java.util.List;

/**
 * Per-primitive inversion strategy: given a desired output value,
 * return one candidate set of input values that would produce it
 * (in slot order). Returns {@code null} when no inversion is possible
 * for the given target.
 *
 * <p>Primitives that have a unique inversion (e.g. lookup-by-key,
 * identity, simple algebraic ops) return the exact answer. Primitives
 * with non-unique inversions (e.g. learnable transforms, ambiguous
 * algebraic splits) return one reasonable candidate by consulting
 * {@link InversionContext} for anchors / reachable values / mandate
 * vocabulary.
 *
 * <p>The carver invokes inversion through
 * {@link Primitive#inversion()}, so each primitive declares its
 * inversion locally; the carver itself contains no primitive-specific
 * branches.
 */
@FunctionalInterface
public interface Inversion {

    List<Value> invert(Value target, InversionContext ctx);

    /** No-op inversion; the carver treats this primitive as a dead end. */
    Inversion NONE = (target, ctx) -> null;
}
