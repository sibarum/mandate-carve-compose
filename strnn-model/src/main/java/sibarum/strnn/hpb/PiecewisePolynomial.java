package sibarum.strnn.hpb;

/**
 * Periodic piecewise-polynomial with double-precision coefficients.
 *
 * <p>Stored as a strictly increasing array of breakpoints over one period
 * {@code [0, period]} (with {@code breakpoints[0] == 0} and
 * {@code breakpoints[last] == period}), and for each piece a polynomial
 * in absolute x: {@code c[0] + c[1]·x + ... + c[d]·x^d}.
 *
 * <p>Evaluation wraps x into {@code [0, period)} via floor-mod, then locates
 * the piece by binary search.
 *
 * <p>Iter 1 uses double-precision throughout. The "exact rational arithmetic"
 * property from §2 of {@code docs/harmonic_piecewise_basis.md} is deferred —
 * it is not load-bearing for the iter-1 falsification tests (F1, F2).
 */
public final class PiecewisePolynomial {

    private final double period;
    private final double[] breakpoints;
    private final double[][] coefficients;

    public PiecewisePolynomial(double period, double[] breakpoints, double[][] coefficients) {
        if (period <= 0.0) throw new IllegalArgumentException("period must be positive: " + period);
        if (breakpoints.length < 2) {
            throw new IllegalArgumentException("need at least 2 breakpoints");
        }
        if (coefficients.length != breakpoints.length - 1) {
            throw new IllegalArgumentException(
                    "coefficients.length (" + coefficients.length + ") must equal "
                            + "breakpoints.length - 1 (" + (breakpoints.length - 1) + ")");
        }
        if (breakpoints[0] != 0.0) {
            throw new IllegalArgumentException("breakpoints[0] must be 0: " + breakpoints[0]);
        }
        double last = breakpoints[breakpoints.length - 1];
        if (Math.abs(last - period) > 1e-12 * period) {
            throw new IllegalArgumentException(
                    "breakpoints[last] (" + last + ") must equal period (" + period + ")");
        }
        for (int i = 1; i < breakpoints.length; i++) {
            if (breakpoints[i] <= breakpoints[i - 1]) {
                throw new IllegalArgumentException(
                        "breakpoints must be strictly increasing at index " + i);
            }
        }
        this.period = period;
        this.breakpoints = breakpoints.clone();
        this.coefficients = new double[coefficients.length][];
        for (int i = 0; i < coefficients.length; i++) {
            this.coefficients[i] = coefficients[i].clone();
        }
    }

    public double period() { return period; }
    public int numPieces() { return coefficients.length; }

    /** Breakpoint at index {@code i}; {@code i ∈ [0, numPieces()]}. */
    public double breakpoint(int i) { return breakpoints[i]; }

    /** Copy of the polynomial coefficients on piece {@code i}, low-degree first. */
    public double[] coefficientsOf(int piece) { return coefficients[piece].clone(); }

    /**
     * Piece index that contains {@code xMod}. Right-continuous at each
     * interior breakpoint: a value exactly at {@code breakpoints[i]} is
     * assigned to piece {@code i} (the piece starting at that breakpoint).
     * Requires {@code 0 <= xMod < period}.
     */
    public int pieceOf(double xMod) {
        int lo = 0;
        int hi = breakpoints.length - 1;
        while (lo + 1 < hi) {
            int mid = (lo + hi) >>> 1;
            if (xMod < breakpoints[mid]) hi = mid;
            else lo = mid;
        }
        return lo;
    }

    /** Periodic evaluation: returns the polynomial value at {@code x mod period}. */
    public double evaluate(double x) {
        double xMod = mod(x, period);
        int piece = pieceOf(xMod);
        return evalPoly(coefficients[piece], xMod);
    }

    /**
     * Antiderivative with respect to x. Same breakpoint partition; each
     * piece's polynomial integrated (degree increased by 1), with the
     * constant of integration chosen for continuity at piece boundaries.
     *
     * <p>The integration starts with {@code A(0) = 0}. If the original
     * function does not have zero mean over one period, the raw
     * antiderivative would have a nonzero value at {@code x = period},
     * breaking the periodicity required for mod-wrap evaluation. To keep
     * the result periodic, a linear drift {@code α·x} is subtracted from
     * every piece, where {@code α = A_raw(period) / period}. This is the
     * "drift-removed" or "circular" antiderivative; it differs from the
     * raw antiderivative by a global linear function and so is equally
     * valid for use in convolution formulas (where the difference of two
     * shifted values cancels any global linear term).
     */
    public PiecewisePolynomial antiderivative() {
        int n = coefficients.length;
        double[][] aCoefs = new double[n][];
        double valueAtBoundary = 0.0;
        for (int i = 0; i < n; i++) {
            double[] c = coefficients[i];
            double[] a = new double[c.length + 1];
            for (int d = 0; d < c.length; d++) a[d + 1] = c[d] / (d + 1);
            double leftBp = breakpoints[i];
            double valueAtLeftWithZeroConst = evalPoly(a, leftBp);
            a[0] = valueAtBoundary - valueAtLeftWithZeroConst;
            aCoefs[i] = a;
            double rightBp = breakpoints[i + 1];
            valueAtBoundary = evalPoly(a, rightBp);
        }
        double drift = valueAtBoundary;
        if (Math.abs(drift) > 1e-12 * Math.max(1.0, period)) {
            double alpha = drift / period;
            for (int i = 0; i < n; i++) {
                double[] a = aCoefs[i];
                if (a.length >= 2) {
                    a[1] -= alpha;
                } else {
                    aCoefs[i] = new double[] { a[0], -alpha };
                }
            }
        }
        return new PiecewisePolynomial(period, breakpoints, aCoefs);
    }

    /**
     * Derivative with respect to x. Same breakpoint partition; each piece's
     * polynomial differentiated coefficient-wise. A piece that is a single
     * constant differentiates to a single zero (kept as length-1 array so
     * downstream piece-by-piece comparisons line up).
     */
    public PiecewisePolynomial derivative() {
        double[][] dCoefs = new double[coefficients.length][];
        for (int i = 0; i < coefficients.length; i++) {
            double[] c = coefficients[i];
            if (c.length <= 1) {
                dCoefs[i] = new double[] { 0.0 };
            } else {
                double[] dc = new double[c.length - 1];
                for (int d = 0; d < dc.length; d++) {
                    dc[d] = (d + 1) * c[d + 1];
                }
                dCoefs[i] = dc;
            }
        }
        return new PiecewisePolynomial(period, breakpoints, dCoefs);
    }

    private static double mod(double x, double m) {
        double r = x % m;
        return r < 0.0 ? r + m : r;
    }

    private static double evalPoly(double[] c, double x) {
        double acc = 0.0;
        for (int d = c.length - 1; d >= 0; d--) {
            acc = acc * x + c[d];
        }
        return acc;
    }
}
