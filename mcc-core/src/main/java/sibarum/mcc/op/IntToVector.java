package sibarum.mcc.op;

import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Trainable indexed embedding: maps an integer index
 * ({@link NumberValue} rounded) to a learnable row of a
 * {@code [vocabSize, dim]} table.
 *
 * <p>The standalone integer-keyed counterpart of {@link sibarum.mcc.embedding.Embed}
 * (which uses string symbols). Use this when categorical inputs arrive
 * as integers (e.g. token IDs, class labels).
 *
 * <p>Backward: only the row at the most-recent index receives a
 * gradient; the input "gradient" is {@code null} since the index is
 * discrete (no continuous slot to propagate to).
 */
public final class IntToVector implements Trainable, Differentiable, Parameterized, Configurable {

    private final int vocabSize;
    private final int dim;
    private final double[] data;   // row-major [vocabSize * dim]
    private int lastIndex = -1;
    private double[] pendingGrad;  // dense [vocabSize * dim], lazily allocated

    public IntToVector(int vocabSize, int dim, long seed) {
        if (vocabSize <= 0 || dim <= 0) {
            throw new IllegalArgumentException(
                    "IntToVector dims must be positive: vocabSize=" + vocabSize + ", dim=" + dim);
        }
        this.vocabSize = vocabSize;
        this.dim = dim;
        this.data = new double[vocabSize * dim];
        Random rng = new Random(seed);
        double bound = Math.sqrt(3.0 / dim);
        for (int i = 0; i < data.length; i++) {
            data[i] = (rng.nextDouble() * 2.0 - 1.0) * bound;
        }
    }

    public int vocabSize() { return vocabSize; }
    public int dim() { return dim; }
    public double[] rawData() { return data; }

    @Override
    public String name() {
        return "int-to-vector";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.NUMBER);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double n = ((NumberValue) inputs.getFirst()).n();
        int idx = (int) Math.round(n);
        if (idx < 0 || idx >= vocabSize) {
            throw new IllegalArgumentException(
                    "IntToVector index " + idx + " out of range [0, " + vocabSize + ")");
        }
        lastIndex = idx;
        double[] row = new double[dim];
        System.arraycopy(data, idx * dim, row, 0, dim);
        return new MatrixValue(row);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastIndex < 0) throw new IllegalStateException("IntToVector backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != dim) {
            throw new IllegalArgumentException("IntToVector gradOutput dim " + g.length + " != " + dim);
        }
        if (pendingGrad == null) pendingGrad = new double[data.length];
        int base = lastIndex * dim;
        for (int i = 0; i < dim; i++) pendingGrad[base + i] += g[i];
        // The integer index has no continuous gradient.
        return java.util.Collections.singletonList(null);
    }

    @Override
    public void step(double lr) {
        if (pendingGrad == null) return;
        for (int i = 0; i < data.length; i++) {
            data[i] -= lr * pendingGrad[i];
        }
        pendingGrad = null;
    }

    @Override
    public Object trainableIdentity() {
        return this;
    }

    @Override
    public Map<String, Object> config() {
        return Map.of("vocabSize", vocabSize, "dim", dim);
    }

    @Override
    public List<NamedTensor> parameters() {
        return List.of(new NamedTensor("table", new int[] { vocabSize, dim }, data.clone()));
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        NamedTensor t = tensors.get("table");
        if (t == null) throw new IllegalArgumentException("IntToVector: missing 'table' tensor");
        if (!Arrays.equals(t.shape(), new int[] { vocabSize, dim })) {
            throw new IllegalArgumentException(
                    "IntToVector 'table' shape mismatch: expected "
                            + Arrays.toString(new int[] { vocabSize, dim })
                            + " got " + Arrays.toString(t.shape()));
        }
        System.arraycopy(t.data(), 0, data, 0, data.length);
    }
}
