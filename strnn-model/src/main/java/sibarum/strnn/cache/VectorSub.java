package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: componentwise vector subtraction. {@code (u, v) -> u − v}.
 *
 * Routed through {@link TotalArithmetic#totalSub} so {@code ±∞ − ±∞ = 0}
 * (same sign) and no NaN can leak.
 */
public final class VectorSub implements Primitive {

    @Override
    public String name() {
        return "vector-sub";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] a = ((MatrixValue) inputs.get(0)).data();
        double[] b = ((MatrixValue) inputs.get(1)).data();
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "VectorSub dim mismatch: " + a.length + " vs " + b.length);
        }
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = TotalArithmetic.totalSub(a[i], b[i]);
        }
        return new MatrixValue(r);
    }
}
