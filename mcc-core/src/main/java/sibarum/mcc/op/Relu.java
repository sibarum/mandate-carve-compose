package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: elementwise ReLU. {@code x_i -> max(0, x_i)}. Backward:
 * {@code dL/dx_i = (x_i > 0) ? dL/dy_i : 0}.
 */
public final class Relu implements Differentiable {

    private double[] lastX;

    @Override
    public String name() {
        return "relu";
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
                throw new IllegalArgumentException("Relu rejects NaN input");
            }
            y[i] = Math.max(0.0, x[i]);
        }
        lastX = x.clone();
        return new MatrixValue(y);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastX == null) throw new IllegalStateException("Relu backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        double[] dx = new double[g.length];
        for (int i = 0; i < g.length; i++) {
            dx[i] = lastX[i] > 0.0 ? g[i] : 0.0;
        }
        return List.of(new MatrixValue(dx));
    }
}
