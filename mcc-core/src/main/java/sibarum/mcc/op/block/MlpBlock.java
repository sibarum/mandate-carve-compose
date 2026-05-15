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
 * Trainable primitive wrapper around {@link Mlp}. Single MatrixValue
 * input → single MatrixValue output. {@link #backward} accepts a
 * target MatrixValue and accumulates gradients via the underlying
 * MLP's MSE backward; {@link #step} applies them.
 *
 * <p>{@link #trainableIdentity} returns the underlying {@link Mlp}
 * instance — multiple {@code MlpBlock}s wrapping the same network
 * share one identity, so the trainer steps the network exactly once
 * per training example.
 */
public final class MlpBlock implements Trainable, Parameterized, Configurable {
    private final Mlp mlp;

    public MlpBlock(int[] sizes, long seed) {
        this.mlp = new Mlp(sizes, seed);
    }

    public MlpBlock(Mlp mlp) {
        this.mlp = mlp;
    }

    public Mlp mlp() {
        return mlp;
    }

    @Override
    public String name() {
        return "mlp-block";
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
        return new MatrixValue(mlp.forward(x));
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (!(gradOutput instanceof MatrixValue mg)) {
            throw new IllegalArgumentException("MlpBlock gradOutput must be MatrixValue");
        }
        double[] dx = mlp.backward(mg.data());
        return List.of(new MatrixValue(dx));
    }

    @Override
    public void step(double lr) {
        mlp.step(lr);
    }

    @Override
    public Object trainableIdentity() {
        return mlp;
    }

    @Override
    public Map<String, Object> config() {
        int[] sizes = mlp.sizes();
        List<Integer> sizesList = new ArrayList<>(sizes.length);
        for (int s : sizes) sizesList.add(s);
        return Map.of("sizes", sizesList);
        // seed intentionally omitted — parameters are restored from blob.
    }

    @Override
    public List<NamedTensor> parameters() {
        List<NamedTensor> out = new ArrayList<>();
        double[][][] ww = mlp.weights();
        double[][] bb = mlp.biases();
        for (int l = 0; l < ww.length; l++) {
            int outDim = ww[l].length;
            int inDim = ww[l][0].length;
            double[] flat = new double[outDim * inDim];
            for (int i = 0; i < outDim; i++) {
                System.arraycopy(ww[l][i], 0, flat, i * inDim, inDim);
            }
            out.add(new NamedTensor("W" + l, new int[] { outDim, inDim }, flat));
            out.add(new NamedTensor("b" + l, new int[] { outDim }, bb[l].clone()));
        }
        return out;
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        double[][][] ww = mlp.weights();
        double[][] bb = mlp.biases();
        for (int l = 0; l < ww.length; l++) {
            NamedTensor wt = tensors.get("W" + l);
            NamedTensor bt = tensors.get("b" + l);
            if (wt == null || bt == null) {
                throw new IllegalArgumentException("MlpBlock: missing W" + l + " or b" + l + " tensor");
            }
            int outDim = ww[l].length;
            int inDim = ww[l][0].length;
            if (!Arrays.equals(wt.shape(), new int[] { outDim, inDim })) {
                throw new IllegalArgumentException(
                        "MlpBlock W" + l + " shape mismatch: expected "
                                + Arrays.toString(new int[] { outDim, inDim })
                                + " got " + Arrays.toString(wt.shape()));
            }
            if (!Arrays.equals(bt.shape(), new int[] { outDim })) {
                throw new IllegalArgumentException("MlpBlock b" + l + " shape mismatch");
            }
            for (int i = 0; i < outDim; i++) {
                System.arraycopy(wt.data(), i * inDim, ww[l][i], 0, inDim);
            }
            System.arraycopy(bt.data(), 0, bb[l], 0, outDim);
        }
    }
}
