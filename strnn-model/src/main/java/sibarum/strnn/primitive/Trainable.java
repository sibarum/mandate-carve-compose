package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;

/**
 * Trainable primitives have learnable parameters. After apply(...) the
 * implementation must remember the inputs and output of the last call so
 * backward(target) can compute gradients. step(lr) applies them.
 */
public interface Trainable extends Primitive {
    void backward(Value target);

    void step(double lr);
}
