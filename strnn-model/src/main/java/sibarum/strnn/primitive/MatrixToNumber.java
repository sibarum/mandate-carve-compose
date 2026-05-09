package sibarum.strnn.primitive;

import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Decodes a 1-dimensional matrix back into a scalar, undoing NumberToMatrix's
 * normalization scale.
 */
public final class MatrixToNumber implements Primitive {
    @Override
    public String name() {
        return "matrix-to-number";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        MatrixValue m = (MatrixValue) inputs.getFirst();
        return new NumberValue(m.data()[0] * NumberToMatrix.SCALE);
    }
}
