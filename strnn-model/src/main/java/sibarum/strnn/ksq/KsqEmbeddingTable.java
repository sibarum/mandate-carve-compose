package sibarum.strnn.ksq;

import java.util.Random;

/**
 * Input embedding table for KSQ: maps a token id in {@code [0, vocab)} to a
 * vector of {@code n} anchor logits in R^n. Softmax over those logits gives
 * the convex anchor coefficients α that build Q(x) = Σ α_i K_i.
 *
 * This is the ONLY learned input-side parameter in the KSQ model. Each call
 * to {@link #lookup(int)} returns the table's internal storage; gradients are
 * accumulated via {@link #accumulateGradient(int, double[])} and consumed by
 * {@link #step(double)} for plain SGD.
 *
 * Initialization: small uniform noise around zero. After softmax, this gives
 * α near the uniform distribution (1/n per anchor), which is a structurally
 * neutral starting point.
 */
public final class KsqEmbeddingTable {

    private final int vocab;
    private final int nAnchors;
    private final double[][] logits;
    private final double[][] gradLogits;

    public KsqEmbeddingTable(int vocab, int nAnchors, long seed) {
        this(vocab, nAnchors, seed, 1.0);
    }

    public KsqEmbeddingTable(int vocab, int nAnchors, long seed, double initBound) {
        if (vocab <= 0) throw new IllegalArgumentException("vocab must be positive: " + vocab);
        if (nAnchors <= 0) throw new IllegalArgumentException("nAnchors must be positive: " + nAnchors);
        if (initBound < 0.0) throw new IllegalArgumentException("initBound must be non-negative: " + initBound);
        this.vocab = vocab;
        this.nAnchors = nAnchors;
        this.logits = new double[vocab][nAnchors];
        this.gradLogits = new double[vocab][nAnchors];

        Random rng = new Random(seed);
        for (int v = 0; v < vocab; v++) {
            for (int a = 0; a < nAnchors; a++) {
                logits[v][a] = (rng.nextDouble() * 2.0 - 1.0) * initBound;
            }
        }
    }

    public int vocab() {
        return vocab;
    }

    public int nAnchors() {
        return nAnchors;
    }

    /**
     * Returns the logit row for {@code tokenId}. The returned array is the
     * table's internal storage; callers MUST NOT mutate it. Use
     * {@link #accumulateGradient} to update.
     */
    public double[] lookup(int tokenId) {
        checkToken(tokenId);
        return logits[tokenId];
    }

    /** Accumulates {@code dLogits} into the gradient slot for {@code tokenId}. */
    public void accumulateGradient(int tokenId, double[] dLogits) {
        checkToken(tokenId);
        if (dLogits.length != nAnchors) {
            throw new IllegalArgumentException(
                    "gradient dim " + dLogits.length + " != nAnchors " + nAnchors);
        }
        for (int a = 0; a < nAnchors; a++) {
            gradLogits[tokenId][a] += dLogits[a];
        }
    }

    /** Applies plain SGD step and zeroes the gradient. */
    public void step(double lr) {
        for (int v = 0; v < vocab; v++) {
            for (int a = 0; a < nAnchors; a++) {
                logits[v][a] -= lr * gradLogits[v][a];
                gradLogits[v][a] = 0.0;
            }
        }
    }

    public void zeroGrad() {
        for (int v = 0; v < vocab; v++) {
            for (int a = 0; a < nAnchors; a++) {
                gradLogits[v][a] = 0.0;
            }
        }
    }

    /** Exposes accumulated gradient for inspection (e.g., the finite-diff gradient check). */
    public double[] gradient(int tokenId) {
        checkToken(tokenId);
        return gradLogits[tokenId];
    }

    private void checkToken(int tokenId) {
        if (tokenId < 0 || tokenId >= vocab) {
            throw new IllegalArgumentException(
                    "tokenId " + tokenId + " out of range [0, " + vocab + ")");
        }
    }
}
