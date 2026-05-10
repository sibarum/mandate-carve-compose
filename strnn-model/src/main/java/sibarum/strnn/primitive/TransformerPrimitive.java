package sibarum.strnn.primitive;

import sibarum.strnn.transformer.Transformer;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Drop-in alternative to MlpPrimitive: same MATRIX(2-dim) -&gt; MATRIX(1-dim)
 * signature, same MlpRole tag, but the underlying learner is a Transformer
 * block. The carver routes through this via the LearnedArithmetic interface
 * without knowing or caring which architecture is inside.
 */
public final class TransformerPrimitive implements Trainable, LearnedArithmetic {
    private final MlpRole role;
    private final Transformer transformer;

    public TransformerPrimitive(MlpRole role, Transformer transformer) {
        this.role = role;
        this.transformer = transformer;
    }

    @Override
    public MlpRole role() {
        return role;
    }

    public Transformer transformer() {
        return transformer;
    }

    @Override
    public String name() {
        return "transformer(" + role.name().toLowerCase() + ")";
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
        if (in.dim() != 2) {
            throw new IllegalArgumentException(
                    "TransformerPrimitive expects a 2-dim composed matrix, got dim=" + in.dim());
        }
        return new MatrixValue(transformer.forward(in.data()));
    }

    @Override
    public void backward(Value target) {
        transformer.backward(((MatrixValue) target).data());
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
