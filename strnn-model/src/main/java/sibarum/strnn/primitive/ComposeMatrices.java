package sibarum.strnn.primitive;

import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Concatenates two matrices into one. v0 binary composition only.
 */
public final class ComposeMatrices implements Primitive {
    @Override
    public String name() {
        return "compose-matrices";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        MatrixValue a = (MatrixValue) inputs.get(0);
        MatrixValue b = (MatrixValue) inputs.get(1);
        double[] combined = new double[a.dim() + b.dim()];
        System.arraycopy(a.data(), 0, combined, 0, a.dim());
        System.arraycopy(b.data(), 0, combined, a.dim(), b.dim());
        return new MatrixValue(combined);
    }
}
