package sibarum.mcc.op.advanced;

/**
 * Periodic piecewise-polynomial with double-precision coefficients.
 *
 * <p>Stored as a strictly increasing array of breakpoints over one period
 * {@code [0, period]} (with {@code breakpoints[0] == 0} and
 * {@code breakpoints[last] == period}), and for each piece a polynomial
 * in absolute x: {@code c[0] + c[1]·x + ... + c[d]·x^d}.
 *
 * <p>Evaluation wraps x into {@code [0, period)} via floor-mod, then
 * locates the piece by binary search. {@link #antiderivative()} applies
 * a drift correction so the result remains periodic even when the input
 * does not have zero mean (the drift cancels in convolution formulas;
 * see {@link SmoothedBasisElement}).
 *
 * <p>Foundation for the harmonic piecewise basis (HPB). Ported from
 * {@code sibarum.strnn.hpb}.
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
    public double breakpoint(int i) { return breakpoints[i]; }
    public double[] coefficientsOf(int piece) { return coefficients[piece].clone(); }

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

    public double evaluate(double x) {
        double xMod = mod(x, period);
        int piece = pieceOf(xMod);
        return evalPoly(coefficients[piece], xMod);
    }

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
