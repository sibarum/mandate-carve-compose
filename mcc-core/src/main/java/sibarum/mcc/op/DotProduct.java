package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (u, v) -> Σ u_i · v_i}. Inner product on two
 * equal-length vectors, returns a scalar. Componentwise products and
 * accumulation go through {@link TotalArithmetic}.
 */
public final class DotProduct implements Primitive {

    @Override
    public String name() {
        return "dot-product";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] a = ((MatrixValue) inputs.get(0)).data();
        double[] b = ((MatrixValue) inputs.get(1)).data();
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "DotProduct dim mismatch: " + a.length + " vs " + b.length);
        }
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s = TotalArithmetic.totalAdd(s, TotalArithmetic.totalMul(a[i], b[i]));
        }
        return new NumberValue(s);
    }
}
