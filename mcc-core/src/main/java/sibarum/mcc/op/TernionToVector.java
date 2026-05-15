package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Reshape: {@code TernionValue → MatrixValue(length 3)}. Pure shape
 * conversion; backward repackages a length-3 {@link MatrixValue}
 * gradient as a {@link TernionValue}.
 */
public final class TernionToVector implements Differentiable {

    @Override
    public String name() {
        return "ternion-to-vector";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TERNION);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TernionValue t = (TernionValue) inputs.getFirst();
        return new MatrixValue(new double[] { t.x(), t.y(), t.z() });
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != 3) {
            throw new IllegalArgumentException("TernionToVector gradOutput length must be 3");
        }
        return List.of(new TernionValue(g[0], g[1], g[2]));
    }
}
