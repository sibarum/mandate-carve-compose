package sibarum.mcc.value;

import java.util.Arrays;

/**
 * Three-component value. Domain-neutral container — the components may
 * represent a 3D Euclidean vector, an RGB triple, three independent
 * scalars, etc. Interpretation belongs to the consuming primitive.
 */
public record TernionValue(double x, double y, double z) implements Value {

    public TernionValue(double[] xyz) {
        this(checkLength(xyz)[0], xyz[1], xyz[2]);
    }

    private static double[] checkLength(double[] xyz) {
        if (xyz.length != 3) {
            throw new IllegalArgumentException("TernionValue requires length 3, got " + xyz.length);
        }
        return xyz;
    }

    @Override
    public ValueType type() {
        return ValueType.TERNION;
    }

    public double[] toArray() {
        return new double[] { x, y, z };
    }

    @Override
    public String toString() {
        return "TernionValue" + Arrays.toString(toArray());
    }
}
