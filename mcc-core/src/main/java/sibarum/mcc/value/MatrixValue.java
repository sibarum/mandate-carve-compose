package sibarum.mcc.value;

import java.util.Arrays;

/**
 * Legacy 1-D vector type carried over from strnn. New primitives that
 * need n-D data should target {@link TensorValue}; {@code MatrixValue}
 * is kept as the workhorse 1-D type to avoid touching pervasive
 * existing call sites.
 */
public record MatrixValue(double[] data) implements Value {
    public MatrixValue {
        data = data.clone();
    }

    @Override
    public ValueType type() {
        return ValueType.MATRIX;
    }

    public int dim() {
        return data.length;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof MatrixValue(double[] data1) && Arrays.equals(data, data1);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "MatrixValue" + Arrays.toString(data);
    }
}
