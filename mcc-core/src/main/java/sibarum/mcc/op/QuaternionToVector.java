package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.QuaternionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Reshape: {@code QuaternionValue → MatrixValue(length 4)}. Pure shape
 * conversion; backward repackages a length-4 {@link MatrixValue}
 * gradient as a {@link QuaternionValue}.
 */
public final class QuaternionToVector implements Differentiable {

    @Override
    public String name() {
        return "quaternion-to-vector";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.QUATERNION);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        QuaternionValue q = (QuaternionValue) inputs.getFirst();
        return new MatrixValue(q.toArray());
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != 4) {
            throw new IllegalArgumentException("QuaternionToVector gradOutput length must be 4");
        }
        return List.of(new QuaternionValue(g));
    }
}
