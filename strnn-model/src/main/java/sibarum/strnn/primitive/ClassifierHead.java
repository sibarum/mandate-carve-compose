package sibarum.strnn.primitive;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Trainable head: {@code MatrixValue (features) -> MatrixValue (logits)}.
 *
 * Wraps an {@link Mlp} of arbitrary shape. Used as the per-task classifier
 * for Part 1's heads (span tagger, per-EntityType classifier, per-RelationType
 * classifier). The output dimension is whatever the wrapped MLP produces —
 * use dim=1 for binary classification, dim=N for N-way multi-class.
 *
 * Two ClassifierHead instances sharing the same Mlp share weights and report
 * the same {@link #trainableIdentity}.
 *
 * Loss is MSE on the raw output. For binary classification, train against
 * targets in {0, 1} and threshold the output at 0.5 at inference; for
 * multi-class, train against one-hot targets and take argmax at inference.
 * Sigmoid / softmax + cross-entropy would be more principled but MSE on
 * scaled targets is the convention this framework uses throughout.
 */
public final class ClassifierHead implements Trainable {

    private final String name;
    private final Mlp mlp;
    private double[] lastInputGradient;

    public ClassifierHead(String name, Mlp mlp) {
        this.name = name;
        this.mlp = mlp;
    }

    public Mlp mlp() {
        return mlp;
    }

    /**
     * Gradient of the loss w.r.t. the input from the most recent call to
     * {@link #backward}. Use this to propagate gradients further upstream
     * (e.g. into an embedding table that produced the input vector).
     * Returns {@code null} if {@link #backward} has not been called yet.
     */
    public double[] inputGradient() {
        return lastInputGradient;
    }

    @Override
    public String name() {
        return "classifier-head(" + name + ")";
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
        MatrixValue in = (MatrixValue) inputs.getFirst();
        if (in.dim() != mlp.inputDim()) {
            throw new IllegalArgumentException(
                    "ClassifierHead '" + name + "' expects dim=" + mlp.inputDim()
                            + ", got dim=" + in.dim());
        }
        return new MatrixValue(mlp.forward(in.data()));
    }

    @Override
    public void backward(Value target) {
        MatrixValue t = (MatrixValue) target;
        if (t.dim() != mlp.outputDim()) {
            throw new IllegalArgumentException(
                    "ClassifierHead '" + name + "' expects target dim=" + mlp.outputDim()
                            + ", got dim=" + t.dim());
        }
        lastInputGradient = mlp.backward(t.data());
    }

    @Override
    public void step(double lr) {
        mlp.step(lr);
    }

    @Override
    public Object trainableIdentity() {
        return mlp;
    }
}
