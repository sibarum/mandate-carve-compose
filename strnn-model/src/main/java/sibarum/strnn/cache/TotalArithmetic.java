package sibarum.strnn.cache;

/**
 * Total, sign-preserving arithmetic on {@code double}, for component-wise use
 * inside vector primitives. Every operation is defined for every pair of
 * inputs except NaN — NaN is an algebra-breaking value and is rejected at the
 * boundary so it cannot propagate.
 *
 * The encoding flattens the omega tower: {@code 2/0} and {@code 5/0} both map
 * to {@code +∞}, and {@code ω·ω} maps to {@code ±∞} rather than {@code ω²}.
 * Information about the rung within the omega/zero tower is lost; this is the
 * accepted cost of collapsing to a single {@code double} per component.
 *
 * IEEE 754 already gives correct totals for most paths (notably
 * {@code finite/0 = sign·∞} and {@code 0/finite = 0} and {@code finite/∞ = 0}).
 * The cases overridden here are exactly the ones IEEE produces NaN for:
 *
 * <pre>
 *   0 · ±∞   →  ±1   (sign of the infinity)
 *   ±∞ · 0   →  ±1
 *   +∞ + −∞  →   0
 *   −∞ + +∞  →   0
 *   +∞ − +∞  →   0
 *   −∞ − −∞  →   0
 *   0 / 0    →   1
 *   ±∞ / ±∞  →  sign(a)·sign(b)
 * </pre>
 */
public final class TotalArithmetic {

    private TotalArithmetic() {
    }

    public static double totalAdd(double a, double b) {
        guardNaN(a, b);
        if (Double.isInfinite(a) && Double.isInfinite(b) && Math.signum(a) != Math.signum(b)) {
            return 0.0;
        }
        return a + b;
    }

    public static double totalSub(double a, double b) {
        guardNaN(a, b);
        if (Double.isInfinite(a) && Double.isInfinite(b) && Math.signum(a) == Math.signum(b)) {
            return 0.0;
        }
        return a - b;
    }

    public static double totalMul(double a, double b) {
        guardNaN(a, b);
        if (a == 0.0 && Double.isInfinite(b)) return Math.signum(b);
        if (Double.isInfinite(a) && b == 0.0) return Math.signum(a);
        return a * b;
    }

    public static double totalDiv(double a, double b) {
        guardNaN(a, b);
        if (a == 0.0 && b == 0.0) return 1.0;
        if (Double.isInfinite(a) && Double.isInfinite(b)) {
            return Math.signum(a) * Math.signum(b);
        }
        return a / b;
    }

    private static void guardNaN(double a, double b) {
        if (Double.isNaN(a) || Double.isNaN(b)) {
            throw new IllegalArgumentException(
                    "NaN input rejected by total arithmetic: a=" + a + ", b=" + b);
        }
    }
}
