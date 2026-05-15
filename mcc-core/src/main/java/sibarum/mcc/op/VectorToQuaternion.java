package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.QuaternionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Reshape: {@code MatrixValue(length 4) → QuaternionValue}. Pure shape
 * conversion; backward passes the gradient straight back as a
 * {@link MatrixValue}.
 */
public final class VectorToQuaternion implements Differentiable {

    @Override
    public String name() {
        return "vector-to-quaternion";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.QUATERNION;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length != 4) {
            throw new IllegalArgumentException("VectorToQuaternion requires length 4, got " + x.length);
        }
        return new QuaternionValue(x[0], x[1], x[2], x[3]);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        QuaternionValue g = (QuaternionValue) gradOutput;
        return List.of(new MatrixValue(g.toArray()));
    }
}
