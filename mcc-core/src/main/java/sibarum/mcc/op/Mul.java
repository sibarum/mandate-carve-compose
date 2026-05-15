package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: componentwise (Hadamard) vector multiplication.
 * {@code (u, v) -> u ⊙ v}. Backward: {@code dL/du = dL/dy ⊙ v,
 * dL/dv = dL/dy ⊙ u}.
 */
public final class Mul implements Differentiable {

    private double[] lastA;
    private double[] lastB;

    @Override
    public String name() {
        return "mul";
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
                    "Mul dim mismatch: " + a.length + " vs " + b.length);
        }
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = TotalArithmetic.totalMul(a[i], b[i]);
        }
        lastA = a.clone();
        lastB = b.clone();
        return new MatrixValue(r);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastA == null) throw new IllegalStateException("Mul backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != lastA.length) {
            throw new IllegalArgumentException(
                    "Mul gradOutput dim " + g.length + " != " + lastA.length);
        }
        double[] dA = new double[g.length];
        double[] dB = new double[g.length];
        for (int i = 0; i < g.length; i++) {
            dA[i] = g[i] * lastB[i];
            dB[i] = g[i] * lastA[i];
        }
        return List.of(new MatrixValue(dA), new MatrixValue(dB));
    }
}
