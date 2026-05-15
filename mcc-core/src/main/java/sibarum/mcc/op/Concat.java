package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: concatenate two vectors end-to-end.
 * {@code (u ∈ ℝⁿ, v ∈ ℝᵐ) -> ℝⁿ⁺ᵐ}.
 *
 * <p>Backward: split {@code gradOut[0:n]} → {@code dL/du},
 * {@code gradOut[n:n+m]} → {@code dL/dv}.
 */
public final class Concat implements Differentiable {

    private int lastN;
    private int lastM;

    @Override
    public String name() {
        return "concat";
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
        double[] out = new double[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        lastN = a.length;
        lastM = b.length;
        return new MatrixValue(out);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastN < 0) throw new IllegalStateException("Concat backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != lastN + lastM) {
            throw new IllegalArgumentException("Concat gradOutput dim mismatch");
        }
        double[] dU = new double[lastN];
        double[] dV = new double[lastM];
        System.arraycopy(g, 0, dU, 0, lastN);
        System.arraycopy(g, lastN, dV, 0, lastM);
        return List.of(new MatrixValue(dU), new MatrixValue(dV));
    }
}
