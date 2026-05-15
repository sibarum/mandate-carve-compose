package sibarum.mcc.primitive;

/**
 * Trainable primitives have learnable parameters. They extend
 * {@link Differentiable}: a gradient flowing in through
 * {@link Differentiable#backward} both produces input-slot gradients
 * and accumulates this primitive's parameter gradients into its
 * internal buffers. {@link #step} then applies the accumulator with
 * the given learning rate and clears it.
 *
 * <p>{@link #trainableIdentity} returns the underlying learnable
 * component (e.g. an MLP instance). Two {@code Trainable} primitives
 * that share the same underlying network return the same identity
 * object — trainers use this for IdentityHashMap-keyed dedup so
 * {@code step()} is called once per unique network per training
 * example.
 */
public interface Trainable extends Differentiable {
    void step(double lr);

    Object trainableIdentity();
}
