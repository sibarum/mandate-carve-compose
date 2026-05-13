package sibarum.strnn.ksq;

import java.util.Random;

/**
 * Output head for KSQ: linear map β ∈ R^n → logits ∈ R^outDim. The second
 * (and final) learned parameter in the model.
 *
 * Standard hand-rolled linear layer: weights W[outDim][n], biases b[outDim],
 * accumulated gradients, SGD step.
 */
public final class KsqOutputHead {

    private final int outDim;
    private final int nAnchors;
    private final double[][] weights;
    private final double[] biases;
    private final double[][] gradWeights;
    private final double[] gradBiases;

    private double[] cachedBeta;

    public KsqOutputHead(int outDim, int nAnchors, long seed) {
        if (outDim <= 0) throw new IllegalArgumentException("outDim must be positive: " + outDim);
        if (nAnchors <= 0) throw new IllegalArgumentException("nAnchors must be positive: " + nAnchors);
        this.outDim = outDim;
        this.nAnchors = nAnchors;
        this.weights = new double[outDim][nAnchors];
        this.biases = new double[outDim];
        this.gradWeights = new double[outDim][nAnchors];
        this.gradBiases = new double[outDim];

        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (nAnchors + outDim));
        for (int k = 0; k < outDim; k++) {
            for (int i = 0; i < nAnchors; i++) {
                weights[k][i] = (rng.nextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    public int outDim() {
        return outDim;
    }

    public int nAnchors() {
        return nAnchors;
    }

    /** Returns logits = W β + b. Caches β for backward. */
    public double[] forward(double[] beta) {
        if (beta.length != nAnchors) {
            throw new IllegalArgumentException(
                    "beta dim " + beta.length + " != nAnchors " + nAnchors);
        }
        cachedBeta = beta.clone();
        double[] logits = new double[outDim];
        for (int k = 0; k < outDim; k++) {
            double s = biases[k];
            for (int i = 0; i < nAnchors; i++) {
                s += weights[k][i] * beta[i];
            }
            logits[k] = s;
        }
        return logits;
    }

    /**
     * Accumulates parameter gradients given {@code dLogits} (∂L/∂logits), and
     * returns {@code dBeta} (∂L/∂β) so the caller can backprop further upstream.
     */
    public double[] backward(double[] dLogits) {
        if (dLogits.length != outDim) {
            throw new IllegalArgumentException(
                    "dLogits dim " + dLogits.length + " != outDim " + outDim);
        }
        if (cachedBeta == null) {
            throw new IllegalStateException("forward must be called before backward");
        }
        for (int k = 0; k < outDim; k++) {
            gradBiases[k] += dLogits[k];
            for (int i = 0; i < nAnchors; i++) {
                gradWeights[k][i] += dLogits[k] * cachedBeta[i];
            }
        }
        double[] dBeta = new double[nAnchors];
        for (int i = 0; i < nAnchors; i++) {
            double s = 0.0;
            for (int k = 0; k < outDim; k++) {
                s += weights[k][i] * dLogits[k];
            }
            dBeta[i] = s;
        }
        return dBeta;
    }

    public void step(double lr) {
        for (int k = 0; k < outDim; k++) {
            biases[k] -= lr * gradBiases[k];
            gradBiases[k] = 0.0;
            for (int i = 0; i < nAnchors; i++) {
                weights[k][i] -= lr * gradWeights[k][i];
                gradWeights[k][i] = 0.0;
            }
        }
    }

    public void zeroGrad() {
        for (int k = 0; k < outDim; k++) {
            gradBiases[k] = 0.0;
            for (int i = 0; i < nAnchors; i++) {
                gradWeights[k][i] = 0.0;
            }
        }
    }

    /** Exposed for the finite-diff gradient check. */
    public double[][] weights() { return weights; }
    public double[] biases() { return biases; }
    public double[][] gradWeights() { return gradWeights; }
    public double[] gradBiases() { return gradBiases; }
}
