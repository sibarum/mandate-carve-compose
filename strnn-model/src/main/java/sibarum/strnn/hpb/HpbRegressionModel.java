package sibarum.strnn.hpb;

import java.util.Random;

/**
 * Regression variant of the HPB model: scalar input → harmonic lift
 * (optionally smoothed) → linear readout → scalar prediction. Trained
 * with half-MSE loss.
 *
 * <p>The basis can be smoothed by any of the kernels supported by
 * {@link SmoothedBasisElement}. The smoothing width is specified as a
 * fraction of each basis element's period: {@code w_k = wFrac · T_k =
 * wFrac / k}. This per-k width keeps the smoothing consistent across
 * frequencies; otherwise a fixed w would over-smooth high-k elements
 * and under-smooth low-k ones.
 *
 * <p>Only the linear readout (W, b) is learned; the basis is fixed.
 * MSE loss has a finite-‖W‖ minimum on well-conditioned regression
 * problems, so SGD converges to a specific point rather than chasing
 * an unbounded ray.
 */
public final class HpbRegressionModel {

    private final int K;
    private final int featDim;
    private final SmoothedBasisElement.Kernel kernel;
    private final double wFrac;

    private final SmoothedBasisElement[] smTri;
    private final SmoothedBasisElement[] smSq;

    private final double[] W;
    private double b;
    private final double[] gW;
    private double gB;

    private double[] cachedFeats;
    private double cachedPred;

    public HpbRegressionModel(int K, SmoothedBasisElement.Kernel kernel, double wFrac, long seed) {
        if (K <= 0) throw new IllegalArgumentException("K must be positive: " + K);
        if (wFrac <= 0.0 && kernel != SmoothedBasisElement.Kernel.DELTA) {
            throw new IllegalArgumentException("wFrac must be positive for non-delta kernel");
        }
        this.K = K;
        this.featDim = 2 * K;
        this.kernel = kernel;
        this.wFrac = wFrac;
        this.smTri = new SmoothedBasisElement[K];
        this.smSq  = new SmoothedBasisElement[K];
        for (int i = 0; i < K; i++) {
            int k = i + 1;
            double T = 1.0 / k;
            double w = T * wFrac;
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq  = HarmonicBasis.sqK(k);
            smTri[i] = wrap(kernel, tri, w);
            smSq[i]  = wrap(kernel, sq,  w);
        }

        this.W = new double[featDim];
        this.gW = new double[featDim];

        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (featDim + 1));
        for (int f = 0; f < featDim; f++) {
            W[f] = (rng.nextDouble() * 2.0 - 1.0) * bound;
        }
    }

    private static SmoothedBasisElement wrap(SmoothedBasisElement.Kernel kernel,
                                             PiecewisePolynomial p, double w) {
        return switch (kernel) {
            case DELTA -> SmoothedBasisElement.delta(p);
            case BOX   -> SmoothedBasisElement.box(p, w);
            case TENT  -> SmoothedBasisElement.tent(p, w);
        };
    }

    public int K() { return K; }
    public int featDim() { return featDim; }
    public SmoothedBasisElement.Kernel kernel() { return kernel; }
    public double wFrac() { return wFrac; }
    public double[] weights() { return W; }
    public double bias() { return b; }

    public double[] lift(double x) {
        double[] f = new double[featDim];
        for (int i = 0; i < K; i++) {
            f[2 * i]     = smTri[i].evaluate(x);
            f[2 * i + 1] = smSq[i].evaluate(x);
        }
        return f;
    }

    public double forward(double x) {
        cachedFeats = lift(x);
        double s = b;
        for (int f = 0; f < featDim; f++) s += W[f] * cachedFeats[f];
        cachedPred = s;
        return cachedPred;
    }

    public double halfMseLoss(double y) {
        double err = cachedPred - y;
        return 0.5 * err * err;
    }

    public void backward(double y) {
        double err = cachedPred - y;
        for (int f = 0; f < featDim; f++) gW[f] += err * cachedFeats[f];
        gB += err;
    }

    public void step(double lr) {
        for (int f = 0; f < featDim; f++) {
            W[f] -= lr * gW[f];
            gW[f] = 0.0;
        }
        b -= lr * gB;
        gB = 0.0;
    }

    public void zeroGrad() {
        for (int f = 0; f < featDim; f++) gW[f] = 0.0;
        gB = 0.0;
    }
}
