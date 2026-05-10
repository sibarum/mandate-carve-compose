package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: componentwise vector addition. {@code (u, v) -> u + v}.
 *
 * Routed through {@link TotalArithmetic#totalAdd} so {@code +∞ + −∞ = 0} and
 * no NaN can leak. Inputs must share dim; mismatched dims throw.
 */
public final class VectorAdd implements Primitive {

    @Override
    public String name() {
        return "vector-add";
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
                    "VectorAdd dim mismatch: " + a.length + " vs " + b.length);
        }
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = TotalArithmetic.totalAdd(a[i], b[i]);
        }
        return new MatrixValue(r);
    }
}
