package sibarum.strnn.ksq.elevator;

import java.util.Random;

/**
 * Input embedding table for elevator KSQ. Unbounded logits (no tanh cap
 * downstream means the table itself doesn't need to be bounded). API
 * mirrors {@code sibarum.strnn.ksq.KsqEmbeddingTable} so demos and
 * gradient-check scaffolding port over with minimal change.
 *
 * <p>Default init bound is 1.0 (small but non-zero); demos can override
 * via the 4-arg constructor for tests that need known initial states.
 */
public final class ElevatorEmbeddingTable {

    private final int vocab;
    private final int nAnchors;
    private final double[][] logits;
    private final double[][] gradLogits;

    public ElevatorEmbeddingTable(int vocab, int nAnchors, long seed) {
        this(vocab, nAnchors, seed, 1.0);
    }

    public ElevatorEmbeddingTable(int vocab, int nAnchors, long seed, double initBound) {
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

    public int vocab() { return vocab; }
    public int nAnchors() { return nAnchors; }

    public double[] lookup(int tokenId) {
        checkToken(tokenId);
        return logits[tokenId];
    }

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
