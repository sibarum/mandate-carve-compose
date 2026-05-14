package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: elementwise ReLU. {@code x_i -> max(0, x_i)}. Forward-only;
 * gradient flow through ReLU is handled by training-time blocks that
 * embed it directly.
 */
public final class Relu implements Primitive {

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
        return new MatrixValue(y);
    }
}
