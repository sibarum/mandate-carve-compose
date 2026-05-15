package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: elementwise hyperbolic tangent. {@code x_i -> tanh(x_i)}.
 * Backward: {@code dL/dx_i = dL/dy_i · (1 − y_i²)}.
 */
public final class Tanh implements Differentiable {

    private double[] lastY;

    @Override
    public String name() {
        return "tanh";
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
            if (Double.isNaN(x[i])) {
                throw new IllegalArgumentException("Tanh rejects NaN input");
            }
            y[i] = Math.tanh(x[i]);
        }
        lastY = y.clone();
        return new MatrixValue(y);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastY == null) throw new IllegalStateException("Tanh backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        double[] dx = new double[g.length];
        for (int i = 0; i < g.length; i++) {
            dx[i] = g[i] * (1.0 - lastY[i] * lastY[i]);
        }
        return List.of(new MatrixValue(dx));
    }
}
