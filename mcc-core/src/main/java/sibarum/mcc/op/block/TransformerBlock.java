package sibarum.mcc.op.block;

import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Trainable primitive wrapper around {@link Transformer}. Single
 * MatrixValue input (flat {@code [seqLen * dIn]}) → single MatrixValue
 * output (length {@code dOut}). {@link #backward} accepts a target
 * MatrixValue; the underlying transformer's MSE backward accumulates
 * gradients; {@link #step} applies them.
 */
public final class TransformerBlock implements Trainable {
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
    public void backward(Value target) {
        if (!(target instanceof MatrixValue mt)) {
            throw new IllegalArgumentException("TransformerBlock target must be MatrixValue");
        }
        transformer.backward(mt.data());
    }

    @Override
    public void step(double lr) {
        transformer.step(lr);
    }

    @Override
    public Object trainableIdentity() {
        return transformer;
    }
}
