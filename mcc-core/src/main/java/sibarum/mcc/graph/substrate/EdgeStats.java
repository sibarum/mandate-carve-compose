package sibarum.mcc.graph.substrate;

/**
 * Running mean of a per-edge success score plus sample count. Used by
 * the (future) carver to rank candidate edges and by a pruner to
 * compare edges sharing a (srcType, dstType) pair.
 */
public final class EdgeStats {
    private double meanScore = 0.5;
    private long samples = 0;
    private boolean pruned = false;

    public synchronized void update(double terminalReward) {
        samples++;
        meanScore += (terminalReward - meanScore) / samples;
    }

    public synchronized double score() {
        return meanScore;
    }

    public synchronized long samples() {
        return samples;
    }

    public synchronized boolean isPruned() {
        return pruned;
    }

    public synchronized void prune() {
        pruned = true;
    }

    /**
     * Hand-bias the mean score. Used to pre-seed preferences before
     * any natural updates have accumulated. Prefer {@link #update} for
     * the normal learning loop.
     */
    public synchronized void setMeanScore(double score) {
        this.meanScore = score;
    }

    /**
     * Clear accumulated stats back to a fresh state. Used by sessions
     * that want to re-bias between runs without rebuilding the
     * substrate.
     */
    public synchronized void reset() {
        this.meanScore = 0.5;
        this.samples = 0;
        this.pruned = false;
    }

    @Override
    public synchronized String toString() {
        return String.format("EdgeStats(score=%.3f, n=%d%s)",
                meanScore, samples, pruned ? ", PRUNED" : "");
    }
}
