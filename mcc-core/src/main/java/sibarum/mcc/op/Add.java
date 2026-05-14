package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: componentwise vector addition. {@code (u, v) -> u + v}.
 *
 * <p>Routed through {@link TotalArithmetic#totalAdd} so
 * {@code +∞ + −∞ = 0} and no NaN can leak. Inputs must share dim;
 * mismatched dims throw.
 */
public final class Add implements Primitive {

    @Override
    public String name() {
        return "add";
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
                    "Add dim mismatch: " + a.length + " vs " + b.length);
        }
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = TotalArithmetic.totalAdd(a[i], b[i]);
        }
        return new MatrixValue(r);
    }
}
