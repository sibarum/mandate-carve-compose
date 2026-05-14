package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: softmax over a vector.
 * {@code logits -> exp(logits − max) / Σ exp(logits − max)}.
 *
 * <p>Numerically stable form: subtracts the per-vector max before
 * exponentiating to avoid overflow. NaN inputs are rejected at the
 * boundary.
 */
public final class Softmax implements Primitive {

    @Override
    public String name() {
        return "softmax";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length == 0) return new MatrixValue(new double[0]);
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (Double.isNaN(v)) {
                throw new IllegalArgumentException("Softmax rejects NaN input");
            }
            if (v > max) max = v;
        }
        double[] out = new double[x.length];
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max);
            sum += out[i];
        }
        if (sum == 0.0) {
            // Pathological: all logits −∞. Fall back to uniform.
            double u = 1.0 / x.length;
            for (int i = 0; i < x.length; i++) out[i] = u;
            return new MatrixValue(out);
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return new MatrixValue(out);
    }
}
