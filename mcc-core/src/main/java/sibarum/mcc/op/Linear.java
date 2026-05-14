package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Trainable affine map: {@code y = W·x (+ b)}. Bias is optional via the
 * {@link #Linear(int, int, boolean, long)} constructor.
 *
 * <p>{@link #apply} caches the input and output. {@link #backward}
 * accepts a target output and accumulates the squared-error gradients
 * {@code (output − target)} into {@code pendingDw}/{@code pendingDb}.
 * {@link #step} applies them and clears the accumulator. This matches
 * the existing strnn-model {@code VectorTransform}/{@code Trainable}
 * contract (target-based backward).
 *
 * <p>Componentwise arithmetic in the forward goes through
 * {@link TotalArithmetic} so pathological inputs (±∞ activations)
 * don't produce NaN.
 */
public final class Linear implements Trainable, Parameterized, Configurable {
    private final int outDim;
    private final int inDim;
    private final boolean withBias;
    private final double[][] w;
    private final double[] b;
    private double[] lastInput;
    private double[] lastOutput;
    private double[][] pendingDw;
    private double[] pendingDb;

    public Linear(int outDim, int inDim, long seed) {
        this(outDim, inDim, true, seed);
    }

    public Linear(int outDim, int inDim, boolean withBias, long seed) {
        if (outDim <= 0 || inDim <= 0) {
            throw new IllegalArgumentException(
                    "dims must be positive: outDim=" + outDim + ", inDim=" + inDim);
        }
        this.outDim = outDim;
        this.inDim = inDim;
        this.withBias = withBias;
        this.w = new double[outDim][inDim];
        this.b = withBias ? new double[outDim] : null;
        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (inDim + outDim));
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                w[i][j] = (rng.nextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    public int outDim() { return outDim; }
    public int inDim() { return inDim; }
    public boolean withBias() { return withBias; }
    public double[][] weights() { return w; }
    public double[] biases() { return b; }

    @Override
    public String name() {
        return withBias ? "linear" : "linear-no-bias";
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
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length != inDim) {
            throw new IllegalArgumentException(
                    "Linear input dim " + x.length + " != expected " + inDim);
        }
        double[] y = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            double s = withBias ? b[i] : 0.0;
            for (int j = 0; j < inDim; j++) {
                s = TotalArithmetic.totalAdd(s, TotalArithmetic.totalMul(w[i][j], x[j]));
            }
            y[i] = s;
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
        if (!(target instanceof MatrixValue mt)) {
            throw new IllegalArgumentException("Linear target must be MatrixValue");
        }
        double[] t = mt.data();
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
        if (withBias) {
            pendingDb = dy.clone();
        }
    }

    @Override
    public void step(double lr) {
        if (pendingDw == null) return;
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                w[i][j] -= lr * pendingDw[i][j];
            }
            if (withBias) {
                b[i] -= lr * pendingDb[i];
            }
        }
        pendingDw = null;
        pendingDb = null;
    }

    @Override
    public Object trainableIdentity() {
        return this;
    }

    @Override
    public Map<String, Object> config() {
        return Map.of(
                "outDim", outDim,
                "inDim", inDim
                // seed intentionally omitted — parameters are restored from blob.
        );
    }

    @Override
    public List<NamedTensor> parameters() {
        List<NamedTensor> out = new ArrayList<>(2);
        double[] flatW = new double[outDim * inDim];
        for (int i = 0; i < outDim; i++) {
            System.arraycopy(w[i], 0, flatW, i * inDim, inDim);
        }
        out.add(new NamedTensor("W", new int[] { outDim, inDim }, flatW));
        if (withBias) {
            out.add(new NamedTensor("b", new int[] { outDim }, b.clone()));
        }
        return out;
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        NamedTensor weights = tensors.get("W");
        if (weights == null) throw new IllegalArgumentException("Linear: missing 'W' tensor");
        if (!Arrays.equals(weights.shape(), new int[] { outDim, inDim })) {
            throw new IllegalArgumentException(
                    "Linear 'W' shape mismatch: expected " + Arrays.toString(new int[] { outDim, inDim })
                            + " got " + Arrays.toString(weights.shape()));
        }
        for (int i = 0; i < outDim; i++) {
            System.arraycopy(weights.data(), i * inDim, w[i], 0, inDim);
        }
        if (withBias) {
            NamedTensor bias = tensors.get("b");
            if (bias == null) throw new IllegalArgumentException("Linear: missing 'b' tensor");
            if (!Arrays.equals(bias.shape(), new int[] { outDim })) {
                throw new IllegalArgumentException("Linear 'b' shape mismatch");
            }
            System.arraycopy(bias.data(), 0, b, 0, outDim);
        }
    }
}
