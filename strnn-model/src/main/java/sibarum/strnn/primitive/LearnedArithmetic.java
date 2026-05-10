package sibarum.strnn.primitive;

/**
 * Marker contract for trainable primitives that play an arithmetic role
 * (ADD or MUL) on a 2-dim composed matrix. The carver uses this to invert
 * candidates without caring whether the underlying learner is an MLP, a
 * transformer block, or anything else with the same signature.
 *
 * Component-agnostic dispatch: BackwardChainingCarver.inferInputs matches
 * against this interface, not against any specific implementation class.
 */
public interface LearnedArithmetic extends Primitive {
    MlpRole role();
}
