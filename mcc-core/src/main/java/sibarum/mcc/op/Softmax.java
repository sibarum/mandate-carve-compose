package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: softmax over a vector.
 * {@code logits -> exp(logits − max) / Σ exp(logits − max)}.
 *
 * <p>Numerically stable form (max-subtract). NaN inputs are rejected.
 *
 * <p>Backward: with {@code y = softmax(x)},
 * {@code dL/dx_i = y_i · (dL/dy_i − Σ_k dL/dy_k · y_k)}.
 */
public final class Softmax implements Differentiable {

    private double[] lastY;

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
        if (x.length == 0) {
            lastY = new double[0];
            return new MatrixValue(lastY);
        }
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
            double u = 1.0 / x.length;
            for (int i = 0; i < x.length; i++) out[i] = u;
        } else {
            for (int i = 0; i < x.length; i++) out[i] /= sum;
        }
        lastY = out.clone();
        return new MatrixValue(out);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastY == null) throw new IllegalStateException("Softmax backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != lastY.length) {
            throw new IllegalArgumentException("Softmax gradOutput dim mismatch");
        }
        double dot = 0.0;
        for (int i = 0; i < g.length; i++) dot += g[i] * lastY[i];
        double[] dx = new double[g.length];
        for (int i = 0; i < g.length; i++) {
            dx[i] = lastY[i] * (g[i] - dot);
        }
        return List.of(new MatrixValue(dx));
    }
}
