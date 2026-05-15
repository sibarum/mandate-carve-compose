package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: matrix-vector product.
 * {@code (A ∈ ℝ^{m×n}, v ∈ ℝⁿ) -> y = A · v ∈ ℝᵐ}.
 *
 * <p>A is a rank-2 {@link TensorValue}; v is a {@link MatrixValue} of
 * length n. Output is a {@link MatrixValue} of length m. The matrix
 * operand typically comes from a {@link Parameter} node.
 *
 * <p>Backward: {@code dA = dy ⊗ v} (outer product),
 * {@code dv = Aᵀ · dy}.
 */
public final class MatVecMul implements Differentiable {

    private double[] lastA;
    private double[] lastV;
    private int lastM;
    private int lastN;

    @Override
    public String name() {
        return "matvecmul";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TENSOR, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TensorValue a = (TensorValue) inputs.get(0);
        MatrixValue v = (MatrixValue) inputs.get(1);
        if (a.rank() != 2) {
            throw new IllegalArgumentException("MatVecMul A must be rank-2, got rank " + a.rank());
        }
        int m = a.dim(0), n = a.dim(1);
        if (v.data().length != n) {
            throw new IllegalArgumentException(
                    "MatVecMul A is " + m + "×" + n + ", v is length " + v.data().length);
        }
        double[] aData = a.data();
        double[] vData = v.data();
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += aData[i * n + j] * vData[j];
            y[i] = s;
        }
        lastA = aData.clone();
        lastV = vData.clone();
        lastM = m; lastN = n;
        return new MatrixValue(y);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastA == null) throw new IllegalStateException("MatVecMul backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != lastM) {
            throw new IllegalArgumentException("MatVecMul gradOutput length " + g.length + " != " + lastM);
        }
        int m = lastM, n = lastN;

        // dA[i,j] = dy[i] · v[j]
        double[] dA = new double[m * n];
        for (int i = 0; i < m; i++) {
            double gi = g[i];
            for (int j = 0; j < n; j++) {
                dA[i * n + j] = gi * lastV[j];
            }
        }

        // dv[j] = Σ_i A[i,j] · dy[i]
        double[] dv = new double[n];
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int i = 0; i < m; i++) s += lastA[i * n + j] * g[i];
            dv[j] = s;
        }

        return List.of(
                new TensorValue(new int[] { m, n }, dA),
                new MatrixValue(dv));
    }
}
