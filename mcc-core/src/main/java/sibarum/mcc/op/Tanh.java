package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: elementwise hyperbolic tangent. {@code x_i -> tanh(x_i)}.
 */
public final class Tanh implements Primitive {

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
        return new MatrixValue(y);
    }
}
