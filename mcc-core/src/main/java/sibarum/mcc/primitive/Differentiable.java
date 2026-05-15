package sibarum.mcc.primitive;

import sibarum.mcc.value.Value;

import java.util.List;

/**
 * Primitives that support gradient backprop. {@link #backward} consumes
 * the gradient of some loss with respect to this primitive's output
 * and returns the gradient with respect to each input slot, in slot
 * order.
 *
 * <p>For stateless ops this lets a gradient flow through during
 * training. For {@link Trainable} primitives the same call additionally
 * accumulates the gradient into internal parameter buffers, ready for
 * a subsequent {@link Trainable#step}.
 *
 * <p>{@code apply} must be called before {@code backward}; implementations
 * cache the input and output of the most recent {@code apply} so the
 * backward pass can use them without re-evaluating.
 *
 * <p>For inputs whose type has no useful continuous gradient (e.g.
 * a {@link sibarum.mcc.value.StringValue} feeding an embedding lookup),
 * the corresponding slot in the returned list should be {@code null}.
 * The trainer treats {@code null} as "do not propagate."
 */
public interface Differentiable extends Primitive {
    List<Value> backward(Value gradOutput);
}
