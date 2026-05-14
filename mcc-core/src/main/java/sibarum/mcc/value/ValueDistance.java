package sibarum.mcc.value;

import java.util.Arrays;

/**
 * Type-aware distance + tolerance-based equality for {@link Value}s.
 * Used by mandate verification to compare expected vs. actual values
 * from a graph execution.
 *
 * <p>Distance semantics:
 * <ul>
 *   <li>Different types → {@link Double#POSITIVE_INFINITY}.</li>
 *   <li>Numeric types ({@code NumberValue}, {@code MatrixValue},
 *       {@code TernionValue}, {@code QuaternionValue},
 *       {@code TensorValue}) use Frobenius / L2 distance over their
 *       components. Shape mismatch (e.g., two tensors with different
 *       shapes) → {@code +∞}.</li>
 *   <li>{@code StringValue} → 0 if equal, 1 otherwise.</li>
 * </ul>
 *
 * <p>{@link #matches} applies the type-appropriate tolerance: numeric
 * types compare Frobenius distance ≤ tolerance; strings compare
 * exact equality.
 */
public final class ValueDistance {
    private ValueDistance() {}

    public static double distance(Value a, Value b) {
        if (a.type() != b.type()) return Double.POSITIVE_INFINITY;
        return switch (a) {
            case NumberValue na -> Math.abs(na.n() - ((NumberValue) b).n());
            case MatrixValue ma -> frobenius(ma.data(), ((MatrixValue) b).data());
            case TernionValue ta -> frobenius(ta.toArray(), ((TernionValue) b).toArray());
            case QuaternionValue qa -> frobenius(qa.toArray(), ((QuaternionValue) b).toArray());
            case TensorValue tva -> tensorDistance(tva, (TensorValue) b);
            case StringValue sa -> sa.s().equals(((StringValue) b).s()) ? 0.0 : 1.0;
        };
    }

    public static boolean matches(Value expected, Value actual, double tolerance) {
        if (expected.type() != actual.type()) return false;
        return switch (expected) {
            case NumberValue ne -> Math.abs(ne.n() - ((NumberValue) actual).n()) <= tolerance;
            case MatrixValue me -> frobenius(me.data(), ((MatrixValue) actual).data()) <= tolerance;
            case TernionValue te -> frobenius(te.toArray(), ((TernionValue) actual).toArray()) <= tolerance;
            case QuaternionValue qe -> frobenius(qe.toArray(), ((QuaternionValue) actual).toArray()) <= tolerance;
            case TensorValue tve -> tensorMatches(tve, (TensorValue) actual, tolerance);
            case StringValue se -> se.s().equals(((StringValue) actual).s());
        };
    }

    private static double tensorDistance(TensorValue a, TensorValue b) {
        if (!Arrays.equals(a.shape(), b.shape())) return Double.POSITIVE_INFINITY;
        return frobenius(a.data(), b.data());
    }

    private static boolean tensorMatches(TensorValue expected, TensorValue actual, double tolerance) {
        if (!Arrays.equals(expected.shape(), actual.shape())) return false;
        return frobenius(expected.data(), actual.data()) <= tolerance;
    }

    private static double frobenius(double[] x, double[] y) {
        if (x.length != y.length) return Double.POSITIVE_INFINITY;
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
}
