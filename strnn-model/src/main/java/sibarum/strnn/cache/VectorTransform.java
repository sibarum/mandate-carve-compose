package sibarum.strnn.cache;

import sibarum.strnn.primitive.Trainable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Random;

/**
 * Primitive: {@code MatrixValue (vector) -> MatrixValue (Wv)}.
 *
 * Learnable single-matrix linear transform. No bias, no activation —
 * deliberately the simplest learnable vector-to-vector op. Adding bias and
 * activation later is a separate primitive choice; keeping this one minimal
 * preserves the abstraction stack.
 *
 * Componentwise arithmetic goes through {@link TotalArithmetic}, so
 * pathological inputs (±∞ embeddings) survive the matrix multiply without
 * producing NaN.
 *
 * Trainable contract: backward(target) treats target as the desired output
 * vector and accumulates ∂L/∂W under squared-error loss. step(lr) applies
 * the SGD update directly to W.
 */
public final class VectorTransform implements Trainable {
    private final int outDim;
    private final int inDim;
    private final double[][] w;
    private double[] lastInput;
    private double[] lastOutput;
    private double[][] pendingDw;

    public VectorTransform(int outDim, int inDim, long seed) {
        if (outDim <= 0 || inDim <= 0) {
            throw new IllegalArgumentException(
                    "dims must be positive: outDim=" + outDim + " inDim=" + inDim);
        }
        this.outDim = outDim;
        this.inDim = inDim;
        this.w = new double[outDim][inDim];
        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (inDim + outDim));
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                w[i][j] = (rng.nextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    public int outDim() {
        return outDim;
    }

    public int inDim() {
        return inDim;
    }

    @Override
    public String name() {
        return "vector-transform";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        MatrixValue m = (MatrixValue) inputs.getFirst();
        double[] x = m.data();
        if (x.length != inDim) {
            throw new IllegalArgumentException(
                    "VectorTransform input dim " + x.length + " != expected " + inDim);
        }
        double[] y = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < inDim; j++) {
                sum = TotalArithmetic.totalAdd(sum,
                        TotalArithmetic.totalMul(w[i][j], x[j]));
            }
            y[i] = sum;
        }
        lastInput = x.clone();
        lastOutput = y.clone();
        return new MatrixValue(y);
    }

    @Override
    public void backward(Value target) {
        if (lastInput == null || lastOutput == null) {
            throw new IllegalStateException("backward called without prior apply");
        }
        if (!(target instanceof MatrixValue m)) {
            throw new IllegalArgumentException("VectorTransform target must be MatrixValue");
        }
        double[] t = m.data();
        if (t.length != outDim) {
            throw new IllegalArgumentException(
                    "target dim " + t.length + " != outDim " + outDim);
        }
        double[] dy = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            dy[i] = lastOutput[i] - t[i];
        }
        pendingDw = new double[outDim][inDim];
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                pendingDw[i][j] = dy[i] * lastInput[j];
            }
        }
    }

    @Override
    public void step(double lr) {
        if (pendingDw == null) return;
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                w[i][j] -= lr * pendingDw[i][j];
            }
        }
        pendingDw = null;
    }

    @Override
    public Object trainableIdentity() {
        return this;
    }
}
