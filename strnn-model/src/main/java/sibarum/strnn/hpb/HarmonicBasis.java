package sibarum.strnn.hpb;

/**
 * Static factory for the harmonic piecewise basis at integer frequency k.
 * Unit-period convention: {@code T = 1/k}. Amplitude fixed at 1
 * (re-weighting is the linear readout's job; see iter-1 plan §Q2).
 *
 * <p>Construction matches §2.1 of {@code docs/harmonic_piecewise_basis.md}:
 * <pre>
 *   tri_k(x) on [0, T]:
 *     [0,    T/4]: 4k·x         (ramps 0 → +1)
 *     [T/4,  T/2]: 2 - 4k·x     (ramps +1 → 0)
 *     [T/2, 3T/4]: 2 - 4k·x     (ramps 0 → -1, slope continues from prior piece)
 *     [3T/4,  T ]: 4k·x - 4     (ramps -1 → 0)
 *
 *   sq_k(x) = d/dx tri_k(x):
 *     [0,    T/4]: +4k
 *     [T/4,  T/2]: -4k
 *     [T/2, 3T/4]: -4k
 *     [3T/4,  T ]: +4k
 * </pre>
 *
 * <p>Pieces 1 and 2 of tri_k share polynomial coefficients — the slope
 * does not change at T/2. The piece partition is kept aligned to sq_k's
 * partition anyway so the iter-1 T1 derivative-pairing check can run
 * piece-by-piece.
 */
public final class HarmonicBasis {

    private HarmonicBasis() {}

    public static PiecewisePolynomial triK(int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
        double T = 1.0 / k;
        double slope = 4.0 * k;
        double[] bp = { 0.0, T / 4.0, T / 2.0, 3.0 * T / 4.0, T };
        double[][] c = new double[][] {
                { 0.0, slope },
                { 2.0, -slope },
                { 2.0, -slope },
                { -4.0, slope }
        };
        return new PiecewisePolynomial(T, bp, c);
    }

    public static PiecewisePolynomial sqK(int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
        double T = 1.0 / k;
        double slope = 4.0 * k;
        double[] bp = { 0.0, T / 4.0, T / 2.0, 3.0 * T / 4.0, T };
        double[][] c = new double[][] {
                { slope },
                { -slope },
                { -slope },
                { slope }
        };
        return new PiecewisePolynomial(T, bp, c);
    }
}
