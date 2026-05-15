package sibarum.mcc.op.block;

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

/**
 * Trainable primitive wrapper around {@link Transformer}. Single
 * MatrixValue input (flat {@code [seqLen * dIn]}) → single MatrixValue
 * output (length {@code dOut}). {@link #backward} accepts a target
 * MatrixValue; the underlying transformer's MSE backward accumulates
 * gradients; {@link #step} applies them.
 */
public final class TransformerBlock implements Trainable, Parameterized, Configurable {
    private final Transformer transformer;

    public TransformerBlock(int seqLen, int dIn, int dModel, int dFf, int dOut, long seed) {
        this.transformer = new Transformer(seqLen, dIn, dModel, dFf, dOut, seed);
    }

    public TransformerBlock(Transformer transformer) {
        this.transformer = transformer;
    }

    public Transformer transformer() {
        return transformer;
    }

    @Override
    public String name() {
        return "transformer-block";
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
        return new MatrixValue(transformer.forward(x));
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (!(gradOutput instanceof MatrixValue mg)) {
            throw new IllegalArgumentException("TransformerBlock gradOutput must be MatrixValue");
        }
        double[] dx = transformer.backward(mg.data());
        return List.of(new MatrixValue(dx));
    }

    @Override
    public void step(double lr) {
        transformer.step(lr);
    }

    @Override
    public Object trainableIdentity() {
        return transformer;
    }

    @Override
    public Map<String, Object> config() {
        return Map.of(
                "seqLen", transformer.seqLen(),
                "dIn", transformer.dIn(),
                "dModel", transformer.dModel(),
                "dFf", transformer.dFf(),
                "dOut", transformer.dOut()
                // seed intentionally omitted — parameters are restored from blob.
        );
    }

    @Override
    public List<NamedTensor> parameters() {
        List<NamedTensor> out = new ArrayList<>(10);
        addMatrix(out, "Win", transformer.win());
        addMatrix(out, "Wq",  transformer.wq());
        addMatrix(out, "Wk",  transformer.wk());
        addMatrix(out, "Wv",  transformer.wv());
        addMatrix(out, "W1",  transformer.w1());
        addVector(out, "b1",  transformer.b1());
        addMatrix(out, "W2",  transformer.w2());
        addVector(out, "b2",  transformer.b2());
        addMatrix(out, "Wo",  transformer.wo());
        addVector(out, "bo",  transformer.bo());
        return out;
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        loadMatrix(tensors, "Win", transformer.win());
        loadMatrix(tensors, "Wq",  transformer.wq());
        loadMatrix(tensors, "Wk",  transformer.wk());
        loadMatrix(tensors, "Wv",  transformer.wv());
        loadMatrix(tensors, "W1",  transformer.w1());
        loadVector(tensors, "b1",  transformer.b1());
        loadMatrix(tensors, "W2",  transformer.w2());
        loadVector(tensors, "b2",  transformer.b2());
        loadMatrix(tensors, "Wo",  transformer.wo());
        loadVector(tensors, "bo",  transformer.bo());
    }

    private static void addMatrix(List<NamedTensor> out, String name, double[][] m) {
        int rows = m.length;
        int cols = m[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(m[i], 0, flat, i * cols, cols);
        }
        out.add(new NamedTensor(name, new int[] { rows, cols }, flat));
    }

    private static void addVector(List<NamedTensor> out, String name, double[] v) {
        out.add(new NamedTensor(name, new int[] { v.length }, v.clone()));
    }

    private static void loadMatrix(Map<String, NamedTensor> tensors, String name, double[][] dst) {
        NamedTensor t = tensors.get(name);
        if (t == null) throw new IllegalArgumentException("TransformerBlock: missing tensor '" + name + "'");
        int rows = dst.length;
        int cols = dst[0].length;
        if (!Arrays.equals(t.shape(), new int[] { rows, cols })) {
            throw new IllegalArgumentException(
                    "TransformerBlock '" + name + "' shape mismatch: expected "
                            + Arrays.toString(new int[] { rows, cols })
                            + " got " + Arrays.toString(t.shape()));
        }
        for (int i = 0; i < rows; i++) {
            System.arraycopy(t.data(), i * cols, dst[i], 0, cols);
        }
    }

    private static void loadVector(Map<String, NamedTensor> tensors, String name, double[] dst) {
        NamedTensor t = tensors.get(name);
        if (t == null) throw new IllegalArgumentException("TransformerBlock: missing tensor '" + name + "'");
        if (!Arrays.equals(t.shape(), new int[] { dst.length })) {
            throw new IllegalArgumentException(
                    "TransformerBlock '" + name + "' shape mismatch");
        }
        System.arraycopy(t.data(), 0, dst, 0, dst.length);
    }
}
