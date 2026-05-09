package sibarum.strnn.primitive;

import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Encodes a scalar number into a 1-dimensional matrix, normalized so the MLPs
 * downstream see values in roughly [0, 1] for the v0 single-digit demo.
 */
public final class NumberToMatrix implements Primitive {
    public static final double SCALE = 10.0;

    @Override
    public String name() {
        return "number-to-matrix";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.NUMBER);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        NumberValue n = (NumberValue) inputs.getFirst();
        return new MatrixValue(new double[]{n.n() / SCALE});
    }
}
