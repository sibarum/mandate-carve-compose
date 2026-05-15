package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Reshape: {@code MatrixValue(length 3) → TernionValue}. Pure shape
 * conversion; backward passes the gradient straight back as a
 * {@link MatrixValue}.
 */
public final class VectorToTernion implements Differentiable {

    @Override
    public String name() {
        return "vector-to-ternion";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TERNION;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length != 3) {
            throw new IllegalArgumentException("VectorToTernion requires length 3, got " + x.length);
        }
        return new TernionValue(x[0], x[1], x[2]);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        TernionValue g = (TernionValue) gradOutput;
        return List.of(new MatrixValue(new double[] { g.x(), g.y(), g.z() }));
    }
}
