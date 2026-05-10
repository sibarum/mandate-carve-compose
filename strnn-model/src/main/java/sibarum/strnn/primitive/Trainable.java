package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;

/**
 * Trainable primitives have learnable parameters. After apply(...) the
 * implementation must remember the inputs and output of the last call so
 * backward(target) can compute gradients. step(lr) applies them.
 *
 * trainableIdentity() returns the underlying learnable component (e.g. an
 * Mlp or Transformer instance). Two Trainable primitives sharing the same
 * underlying network return the same identity object — the trainer uses
 * this for IdentityHashMap-keyed dedup so step() is called once per unique
 * network per example.
 */
public interface Trainable extends Primitive {
    void backward(Value target);

    void step(double lr);

    Object trainableIdentity();
}
