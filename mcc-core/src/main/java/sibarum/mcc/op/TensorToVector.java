package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Reshape: {@code TensorValue → MatrixValue} (flatten). Backward
 * repackages the flat gradient back into the source tensor shape,
 * which is recorded from the most recent {@code apply}.
 */
public final class TensorToVector implements Differentiable {

    private int[] lastShape;

    @Override
    public String name() {
        return "tensor-to-vector";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TENSOR);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TensorValue t = (TensorValue) inputs.getFirst();
        lastShape = t.shape();
        return new MatrixValue(t.data());
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastShape == null) throw new IllegalStateException("TensorToVector backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != TensorValue.volume(lastShape)) {
            throw new IllegalArgumentException("TensorToVector gradOutput dim mismatch");
        }
        return List.of(new TensorValue(lastShape, g));
    }
}
