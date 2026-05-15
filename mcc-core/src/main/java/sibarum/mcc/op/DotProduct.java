package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (u, v) -> Σ u_i · v_i}. Inner product on two
 * equal-length vectors, returns a scalar.
 *
 * <p>Backward: {@code dL/du = gradOut · v}, {@code dL/dv = gradOut · u}.
 */
public final class DotProduct implements Differentiable {

    private double[] lastA;
    private double[] lastB;

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
        lastA = a.clone();
        lastB = b.clone();
        return new NumberValue(s);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastA == null) throw new IllegalStateException("DotProduct backward without prior apply");
        double g = ((NumberValue) gradOutput).n();
        double[] dA = new double[lastA.length];
        double[] dB = new double[lastB.length];
        for (int i = 0; i < lastA.length; i++) {
            dA[i] = g * lastB[i];
            dB[i] = g * lastA[i];
        }
        return List.of(new MatrixValue(dA), new MatrixValue(dB));
    }
}
