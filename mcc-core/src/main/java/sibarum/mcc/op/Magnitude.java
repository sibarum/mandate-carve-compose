package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: L2 norm. {@code u -> √(Σ u_i²)}. Returns a scalar.
 *
 * <p>Uses plain IEEE arithmetic in the accumulation. Infinite components
 * propagate; NaN inputs are rejected at the boundary.
 */
public final class Magnitude implements Primitive {

    @Override
    public String name() {
        return "magnitude";
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
        double[] a = ((MatrixValue) inputs.getFirst()).data();
        double s = 0.0;
        for (double v : a) {
            if (Double.isNaN(v)) {
                throw new IllegalArgumentException("Magnitude rejects NaN input");
            }
            s += v * v;
        }
        return new NumberValue(Math.sqrt(s));
    }
}
