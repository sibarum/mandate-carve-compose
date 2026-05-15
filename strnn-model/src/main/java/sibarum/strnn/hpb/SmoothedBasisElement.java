package sibarum.strnn.hpb;

/**
 * Convolution of a periodic piecewise-polynomial basis element with a
 * compact-support kernel, evaluated by closed-form difference of shifted
 * antiderivatives — no numerical integration.
 *
 * <p>Three kernel choices, all unit-integral and centered on zero:
 * <ul>
 *   <li><b>delta(p)</b>: identity. Returns the original basis.</li>
 *   <li><b>box(p, w)</b>: rectangular pulse, support {@code [-w/2, w/2]},
 *       height {@code 1/w}. Smoothed value:
 *       {@code (F(x + w/2) - F(x - w/2)) / w} where F is the first
 *       antiderivative of the original. Piecewise quadratic if input is
 *       piecewise linear; C^0 at smoothed corners.</li>
 *   <li><b>tent(p, w)</b>: triangular pulse, support {@code [-w/2, w/2]},
 *       peak {@code 2/w} at zero. Equal to {@code box(w/2) * box(w/2)}.
 *       Smoothed value:
 *       {@code (4/w²) · (G(x + w/2) - 2 G(x) + G(x - w/2))}
 *       where G is the second antiderivative of the original. Piecewise
 *       cubic if input is piecewise linear; C^1 at smoothed corners.</li>
 * </ul>
 *
 * <p>Both formulas auto-handle the periodic boundary via the underlying
 * {@link PiecewisePolynomial#evaluate(double)}'s mod-wrap. The
 * antiderivative class guarantees periodicity by drift-removal, so even
 * if the original's antiderivative has nonzero mean (which forces a
 * drift in the second antiderivative), the combination
 * {@code G(x+w/2) - 2 G(x) + G(x-w/2)} is mathematically periodic and
 * the drift-removed evaluation gives the correct value.
 */
public final class SmoothedBasisElement {

    public enum Kernel { DELTA, BOX, TENT }

    private final Kernel kernel;
    private final double w;
    private final PiecewisePolynomial original;
    private final PiecewisePolynomial F;
    private final PiecewisePolynomial G;

    private SmoothedBasisElement(Kernel kernel, double w,
                                 PiecewisePolynomial original,
                                 PiecewisePolynomial F,
                                 PiecewisePolynomial G) {
        this.kernel = kernel;
        this.w = w;
        this.original = original;
        this.F = F;
        this.G = G;
    }

    public static SmoothedBasisElement delta(PiecewisePolynomial p) {
        return new SmoothedBasisElement(Kernel.DELTA, 0.0, p, null, null);
    }

    public static SmoothedBasisElement box(PiecewisePolynomial p, double w) {
        if (w <= 0.0) throw new IllegalArgumentException("width must be positive: " + w);
        return new SmoothedBasisElement(Kernel.BOX, w, p, p.antiderivative(), null);
    }

    public static SmoothedBasisElement tent(PiecewisePolynomial p, double w) {
        if (w <= 0.0) throw new IllegalArgumentException("width must be positive: " + w);
        PiecewisePolynomial first = p.antiderivative();
        return new SmoothedBasisElement(Kernel.TENT, w, p, first, first.antiderivative());
    }

    public Kernel kernel() { return kernel; }
    public double width() { return w; }

    public double evaluate(double x) {
        return switch (kernel) {
            case DELTA -> original.evaluate(x);
            case BOX -> (F.evaluate(x + w / 2.0) - F.evaluate(x - w / 2.0)) / w;
            case TENT -> (4.0 / (w * w)) * (
                    G.evaluate(x + w / 2.0)
                            - 2.0 * G.evaluate(x)
                            + G.evaluate(x - w / 2.0));
        };
    }
}
