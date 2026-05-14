package sibarum.mcc.primitive;

import sibarum.mcc.value.Value;

/**
 * Trainable primitives have learnable parameters.
 *
 * <p>Contract: after {@code apply(...)} the implementation must remember
 * the inputs and output of the last call so {@link #backward} can
 * compute parameter gradients. {@link #step} then applies them with
 * the given learning rate and clears the accumulated gradients.
 *
 * <p>{@link #trainableIdentity} returns the underlying learnable
 * component (e.g. an {@code Mlp} instance). Two {@code Trainable}
 * primitives that share the same underlying network return the same
 * identity object — trainers use this for IdentityHashMap-keyed dedup
 * so {@code step()} is called once per unique network per example.
 */
public interface Trainable extends Primitive {
    void backward(Value target);

    void step(double lr);

    Object trainableIdentity();
}
