package sibarum.mcc.op.advanced;

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
 * <p>Derivative under each kernel is computable analytically:
 * <ul>
 *   <li>delta: {@code d/dx f(x) = f'(x)}</li>
 *   <li>box:   {@code d/dx (f * box) = (f(x+w/2) − f(x-w/2))/w}</li>
 *   <li>tent:  {@code d/dx (f * tent) = (4/w²)(F(x+w/2) − 2F(x) + F(x-w/2))}</li>
 * </ul>
 * The {@link #evaluateDerivative(double)} method returns these closed-form
 * values directly, used by {@link sibarum.mcc.op.HarmonicLift}'s backward.
 *
 * <p>Ported from {@code sibarum.strnn.hpb} with an added derivative API.
 */
public final class SmoothedBasisElement {

    public enum Kernel { DELTA, BOX, TENT }

    private final Kernel kernel;
    private final double w;
    private final PiecewisePolynomial original;
    private final PiecewisePolynomial originalDerivative;
    private final PiecewisePolynomial F;
    private final PiecewisePolynomial G;

    private SmoothedBasisElement(Kernel kernel, double w,
                                 PiecewisePolynomial original,
                                 PiecewisePolynomial originalDerivative,
                                 PiecewisePolynomial F,
                                 PiecewisePolynomial G) {
        this.kernel = kernel;
        this.w = w;
        this.original = original;
        this.originalDerivative = originalDerivative;
        this.F = F;
        this.G = G;
    }

    public static SmoothedBasisElement delta(PiecewisePolynomial p) {
        return new SmoothedBasisElement(Kernel.DELTA, 0.0, p, p.derivative(), null, null);
    }

    public static SmoothedBasisElement box(PiecewisePolynomial p, double w) {
        if (w <= 0.0) throw new IllegalArgumentException("width must be positive: " + w);
        return new SmoothedBasisElement(Kernel.BOX, w, p, null, p.antiderivative(), null);
    }

    public static SmoothedBasisElement tent(PiecewisePolynomial p, double w) {
        if (w <= 0.0) throw new IllegalArgumentException("width must be positive: " + w);
        PiecewisePolynomial first = p.antiderivative();
        return new SmoothedBasisElement(Kernel.TENT, w, p, null, first, first.antiderivative());
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

    /**
     * Returns the derivative of the smoothed basis at x. Closed-form, no
     * numerical estimation. For δ the result is the derivative of the
     * underlying piecewise polynomial (right-continuous at breakpoints);
     * for box and tent the result is smooth everywhere except at the
     * smoothed-piece boundaries (still right-continuous there).
     */
    public double evaluateDerivative(double x) {
        return switch (kernel) {
            case DELTA -> originalDerivative.evaluate(x);
            case BOX -> (original.evaluate(x + w / 2.0) - original.evaluate(x - w / 2.0)) / w;
            case TENT -> (4.0 / (w * w)) * (
                    F.evaluate(x + w / 2.0)
                            - 2.0 * F.evaluate(x)
                            + F.evaluate(x - w / 2.0));
        };
    }
}
