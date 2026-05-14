package sibarum.mcc.value;

import java.util.Arrays;

/**
 * Four-component value. Components are stored in order {@code (a, b, c, d)};
 * the consuming primitive decides the algebraic interpretation —
 * Hamilton quaternions ({@code a + b·i + c·j + d·k}), split quaternions
 * (see {@code op/advanced/SplitQuat} in mcc-core, signature ({@code ++--})),
 * 2×2 matrices flattened, or just four independent scalars.
 */
public record QuaternionValue(double a, double b, double c, double d) implements Value {

    public QuaternionValue(double[] abcd) {
        this(checkLength(abcd)[0], abcd[1], abcd[2], abcd[3]);
    }

    private static double[] checkLength(double[] abcd) {
        if (abcd.length != 4) {
            throw new IllegalArgumentException("QuaternionValue requires length 4, got " + abcd.length);
        }
        return abcd;
    }

    @Override
    public ValueType type() {
        return ValueType.QUATERNION;
    }

    public double[] toArray() {
        return new double[] { a, b, c, d };
    }

    @Override
    public String toString() {
        return "QuaternionValue" + Arrays.toString(toArray());
    }
}
