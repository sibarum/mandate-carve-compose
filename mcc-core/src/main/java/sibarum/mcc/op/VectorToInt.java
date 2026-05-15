package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: argmax. {@code v ∈ ℝⁿ -> NumberValue(index of largest entry)}.
 *
 * <p>Ties are broken by lowest-index-wins.
 *
 * <p>Backward: argmax is a discrete operation; no continuous gradient
 * flows back to the input. {@link #backward} returns {@code null} for
 * the input-gradient slot. This means a graph training through
 * {@code VectorToInt} must place the trainable upstream so that the
 * loss reaches it via a different path (e.g. an auxiliary cross-entropy
 * arm on the logits).
 */
public final class VectorToInt implements Differentiable {

    private int lastDim;

    @Override
    public String name() {
        return "vector-to-int";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length == 0) {
            throw new IllegalArgumentException("VectorToInt requires a non-empty vector");
        }
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) best = i;
        }
        lastDim = x.length;
        return new NumberValue(best);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastDim == 0) throw new IllegalStateException("VectorToInt backward without prior apply");
        // Argmax is non-differentiable; no continuous gradient flows back.
        return java.util.Collections.singletonList(null);
    }
}
