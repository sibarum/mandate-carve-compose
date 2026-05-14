package sibarum.mcc.value;

import java.util.Arrays;

/**
 * n-D shape-tagged tensor. Storage is row-major contiguous in a flat
 * {@code double[]}. New primitives in mcc-core should target
 * {@code TensorValue} so the rank/shape is part of the value rather
 * than implicit.
 *
 * <p>The {@code Tensor} backend is intentionally a plain {@code double[]}
 * for the MVP. A future SPI seam (SIMD, JNI BLAS, etc.) can swap the
 * storage strategy without changing the value type.
 */
public record TensorValue(int[] shape, double[] data) implements Value {

    public TensorValue {
        if (shape == null) {
            throw new IllegalArgumentException("shape must not be null");
        }
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        for (int d : shape) {
            if (d < 0) throw new IllegalArgumentException("shape dimensions must be non-negative: " + Arrays.toString(shape));
        }
        int expected = volume(shape);
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " does not match shape volume " + expected
                            + " for shape " + Arrays.toString(shape));
        }
        shape = shape.clone();
        data = data.clone();
    }

    @Override
    public ValueType type() {
        return ValueType.TENSOR;
    }

    public int rank() {
        return shape.length;
    }

    public int size() {
        return data.length;
    }

    public int dim(int axis) {
        return shape[axis];
    }

    public static int volume(int[] shape) {
        if (shape.length == 0) return 1;
        long v = 1;
        for (int d : shape) v *= d;
        if (v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("shape volume exceeds int range: " + Arrays.toString(shape));
        }
        return (int) v;
    }

    /** Scalar (rank-0) tensor. */
    public static TensorValue scalar(double v) {
        return new TensorValue(new int[0], new double[] { v });
    }

    /** Rank-1 (vector) convenience constructor. */
    public static TensorValue vector(double... data) {
        return new TensorValue(new int[] { data.length }, data);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof TensorValue(int[] s, double[] d)
                && Arrays.equals(shape, s)
                && Arrays.equals(data, d);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(shape) + Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "TensorValue(shape=" + Arrays.toString(shape) + ", data=" + Arrays.toString(data) + ")";
    }
}
