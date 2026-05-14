package sibarum.mcc.op.advanced;

/**
 * Tiny 2×2 real matrix helpers. Originally KSQ's hot path; lifted into
 * mcc-core for any primitive that wants a fast 2×2 matmul.
 *
 * <p>Matrices are stored as {@code double[2][2]}, row-major. No
 * allocation in the inner-loop hot path beyond the result matrix
 * itself.
 */
public final class Mat2 {

    private Mat2() {}

    /** Returns the 2×2 identity. */
    public static double[][] identity() {
        return new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } };
    }

    /** Returns the 2×2 zero matrix. */
    public static double[][] zero() {
        return new double[2][2];
    }

    /** Returns a deep copy of {@code m}. */
    public static double[][] copy(double[][] m) {
        return new double[][] { { m[0][0], m[0][1] }, { m[1][0], m[1][1] } };
    }

    /** Returns {@code a · b} as a fresh matrix. Non-commutative. */
    public static double[][] mul(double[][] a, double[][] b) {
        double[][] c = new double[2][2];
        c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
        c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
        c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
        c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
        return c;
    }

    /** Returns {@code m^T} as a fresh matrix. */
    public static double[][] transpose(double[][] m) {
        return new double[][] { { m[0][0], m[1][0] }, { m[0][1], m[1][1] } };
    }

    /** Frobenius inner product: sum of elementwise products. */
    public static double frobInner(double[][] a, double[][] b) {
        return a[0][0] * b[0][0] + a[0][1] * b[0][1]
                + a[1][0] * b[1][0] + a[1][1] * b[1][1];
    }

    /** In-place: {@code dst += scale * src}. */
    public static void addScaledInPlace(double[][] dst, double[][] src, double scale) {
        dst[0][0] += scale * src[0][0];
        dst[0][1] += scale * src[0][1];
        dst[1][0] += scale * src[1][0];
        dst[1][1] += scale * src[1][1];
    }
}
