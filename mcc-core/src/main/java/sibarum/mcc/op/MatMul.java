package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: matrix multiplication.
 * {@code (A ∈ ℝ^{m×k}, B ∈ ℝ^{k×n}) -> C = A · B ∈ ℝ^{m×n}}.
 *
 * <p>Both operands are rank-2 {@link TensorValue}s. The matrix operand
 * typically comes from a {@link Parameter} node in a training graph;
 * MatMul itself is stateless.
 *
 * <p>Backward: {@code dA = dC · Bᵀ}, {@code dB = Aᵀ · dC}.
 */
public final class MatMul implements Differentiable {

    private double[] lastA;
    private double[] lastB;
    private int lastM;
    private int lastK;
    private int lastN;

    @Override
    public String name() {
        return "matmul";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TENSOR, ValueType.TENSOR);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TENSOR;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TensorValue a = (TensorValue) inputs.get(0);
        TensorValue b = (TensorValue) inputs.get(1);
        if (a.rank() != 2 || b.rank() != 2) {
            throw new IllegalArgumentException(
                    "MatMul requires rank-2 tensors, got A rank=" + a.rank() + ", B rank=" + b.rank());
        }
        int m = a.dim(0), k = a.dim(1), n = b.dim(1);
        if (b.dim(0) != k) {
            throw new IllegalArgumentException(
                    "MatMul inner dims must match: A is " + m + "×" + k + ", B is "
                            + b.dim(0) + "×" + n);
        }
        double[] aData = a.data();
        double[] bData = b.data();
        double[] cData = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int kk = 0; kk < k; kk++) {
                double aik = aData[i * k + kk];
                for (int j = 0; j < n; j++) {
                    cData[i * n + j] += aik * bData[kk * n + j];
                }
            }
        }
        lastA = aData.clone();
        lastB = bData.clone();
        lastM = m; lastK = k; lastN = n;
        return new TensorValue(new int[] { m, n }, cData);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastA == null) throw new IllegalStateException("MatMul backward without prior apply");
        TensorValue g = (TensorValue) gradOutput;
        if (g.rank() != 2 || g.dim(0) != lastM || g.dim(1) != lastN) {
            throw new IllegalArgumentException("MatMul gradOutput shape mismatch");
        }
        double[] gData = g.data();
        int m = lastM, k = lastK, n = lastN;

        // dA[i,kk] = Σ_j dC[i,j] · B[kk,j]
        double[] dA = new double[m * k];
        for (int i = 0; i < m; i++) {
            for (int kk = 0; kk < k; kk++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += gData[i * n + j] * lastB[kk * n + j];
                }
                dA[i * k + kk] = s;
            }
        }

        // dB[kk,j] = Σ_i A[i,kk] · dC[i,j]
        double[] dB = new double[k * n];
        for (int kk = 0; kk < k; kk++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = 0; i < m; i++) {
                    s += lastA[i * k + kk] * gData[i * n + j];
                }
                dB[kk * n + j] = s;
            }
        }

        return List.of(
                new TensorValue(new int[] { m, k }, dA),
                new TensorValue(new int[] { k, n }, dB));
    }
}
