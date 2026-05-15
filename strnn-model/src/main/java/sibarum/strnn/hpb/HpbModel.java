package sibarum.strnn.hpb;

import java.util.Random;

/**
 * Harmonic Piecewise Basis model: scalar input → harmonic lift → linear readout.
 *
 * <p>Forward pipeline:
 * <pre>
 *   features = [tri_1(x), sq_1(x), tri_2(x), sq_2(x), ..., tri_K(x), sq_K(x)] ∈ R^{2K}
 *   logits   = W · features + b ∈ R^outDim
 * </pre>
 *
 * <p>The basis is fixed (δ kernel, raw piecewise-linear). Only W and b are
 * learned. This is the iter-1 architecture: smallest unit that tests the
 * doc's F2 falsification claim (exact-rational basin solves XOR).
 *
 * <p>Backward under softmax-cross-entropy. The basis itself has no
 * parameters, so the backward pass terminates at the lift — there is no
 * dx gradient consumer in iter 1.
 */
public final class HpbModel {

    private final int K;
    private final int outDim;
    private final int featDim;

    private final PiecewisePolynomial[] tri;
    private final PiecewisePolynomial[] sq;

    private final double[][] weights;
    private final double[] biases;
    private final double[][] gradWeights;
    private final double[] gradBiases;

    private double[] cachedFeatures;
    private double[] cachedLogits;

    public HpbModel(int K, int outDim, long seed) {
        if (K <= 0) throw new IllegalArgumentException("K must be positive: " + K);
        if (outDim <= 0) throw new IllegalArgumentException("outDim must be positive: " + outDim);
        this.K = K;
        this.outDim = outDim;
        this.featDim = 2 * K;
        this.tri = new PiecewisePolynomial[K];
        this.sq = new PiecewisePolynomial[K];
        for (int i = 0; i < K; i++) {
            tri[i] = HarmonicBasis.triK(i + 1);
            sq[i] = HarmonicBasis.sqK(i + 1);
        }
        this.weights = new double[outDim][featDim];
        this.biases = new double[outDim];
        this.gradWeights = new double[outDim][featDim];
        this.gradBiases = new double[outDim];

        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (featDim + outDim));
        for (int o = 0; o < outDim; o++) {
            for (int f = 0; f < featDim; f++) {
                weights[o][f] = (rng.nextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    public int K() { return K; }
    public int outDim() { return outDim; }
    public int featDim() { return featDim; }

    public double[][] weights() { return weights; }
    public double[] biases() { return biases; }
    public double[][] gradWeights() { return gradWeights; }
    public double[] gradBiases() { return gradBiases; }

    /** Harmonic lift at scalar x. Returns a fresh array. */
    public double[] lift(double x) {
        double[] f = new double[featDim];
        for (int i = 0; i < K; i++) {
            f[2 * i]     = tri[i].evaluate(x);
            f[2 * i + 1] = sq[i].evaluate(x);
        }
        return f;
    }

    public double[] forward(double x) {
        cachedFeatures = lift(x);
        cachedLogits = new double[outDim];
        for (int o = 0; o < outDim; o++) {
            double s = biases[o];
            for (int f = 0; f < featDim; f++) {
                s += weights[o][f] * cachedFeatures[f];
            }
            cachedLogits[o] = s;
        }
        return cachedLogits.clone();
    }

    public double crossEntropyLoss(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");
        if (target < 0 || target >= cachedLogits.length) {
            throw new IllegalArgumentException("target out of range: " + target);
        }
        double m = Double.NEGATIVE_INFINITY;
        for (double v : cachedLogits) if (v > m) m = v;
        double sumExp = 0.0;
        for (double v : cachedLogits) sumExp += Math.exp(v - m);
        return Math.log(sumExp) + m - cachedLogits[target];
    }

    public void backward(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");
        double[] dLogits = softmax(cachedLogits);
        dLogits[target] -= 1.0;
        for (int o = 0; o < outDim; o++) {
            gradBiases[o] += dLogits[o];
            for (int f = 0; f < featDim; f++) {
                gradWeights[o][f] += dLogits[o] * cachedFeatures[f];
            }
        }
    }

    public void step(double lr) {
        for (int o = 0; o < outDim; o++) {
            biases[o] -= lr * gradBiases[o];
            gradBiases[o] = 0.0;
            for (int f = 0; f < featDim; f++) {
                weights[o][f] -= lr * gradWeights[o][f];
                gradWeights[o][f] = 0.0;
            }
        }
    }

    public void zeroGrad() {
        for (int o = 0; o < outDim; o++) {
            gradBiases[o] = 0.0;
            for (int f = 0; f < featDim; f++) {
                gradWeights[o][f] = 0.0;
            }
        }
    }

    private static double[] softmax(double[] logits) {
        int n = logits.length;
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logits) if (v > m) m = v;
        double[] out = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            out[i] = Math.exp(logits[i] - m);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
        return out;
    }
}
