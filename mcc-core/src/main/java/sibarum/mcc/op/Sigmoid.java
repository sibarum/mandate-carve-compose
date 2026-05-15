package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: elementwise sigmoid. {@code x_i -> 1 / (1 + exp(−x_i))}.
 * Numerically stable on both tails. Backward:
 * {@code dL/dx_i = dL/dy_i · y_i · (1 − y_i)}.
 */
public final class Sigmoid implements Differentiable {

    private double[] lastY;

    @Override
    public String name() {
        return "sigmoid";
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
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double v = x[i];
            if (Double.isNaN(v)) {
                throw new IllegalArgumentException("Sigmoid rejects NaN input");
            }
            if (v >= 0.0) {
                double e = Math.exp(-v);
                y[i] = 1.0 / (1.0 + e);
            } else {
                double e = Math.exp(v);
                y[i] = e / (1.0 + e);
            }
        }
        lastY = y.clone();
        return new MatrixValue(y);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastY == null) throw new IllegalStateException("Sigmoid backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        double[] dx = new double[g.length];
        for (int i = 0; i < g.length; i++) {
            dx[i] = g[i] * lastY[i] * (1.0 - lastY[i]);
        }
        return List.of(new MatrixValue(dx));
    }
}
