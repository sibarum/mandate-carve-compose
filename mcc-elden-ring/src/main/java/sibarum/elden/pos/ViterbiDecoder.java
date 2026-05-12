package sibarum.elden.pos;

/**
 * Viterbi dynamic-programming decode over a sequence of per-position emission
 * scores plus a tag-bigram transition log-prob matrix.
 *
 * Emissions are the trainer's MLP logits (raw, no softmax). Transitions are
 * smoothed log-probabilities computed by {@link PosTransitions}. The single
 * tuning knob is {@code transWeight} — a scalar multiplier on the transition
 * contribution so the relative magnitude of emission vs transition signal can
 * be balanced.
 *
 * At {@code transWeight == 0} this reduces to position-independent greedy
 * argmax. As {@code transWeight} grows, the decoder cares more about producing
 * a plausible *sequence* of tags than about any individual token's emission
 * score — until eventually the most-likely-sequence prior dominates and the
 * tagger ignores the input.
 */
public final class ViterbiDecoder {

    private ViterbiDecoder() {}

    /** Convenience overload — POS transitions. */
    public static int[] decode(double[][] emissions, PosTransitions trans, double transWeight) {
        return decode(emissions, trans.logTrans, trans.logInitial, transWeight);
    }

    /**
     * @param emissions    [T][K] per-position emission scores (MLP logits)
     * @param logTrans     [K][K] log-probability transitions (prev -> next).
     *                     Use {@link Double#NEGATIVE_INFINITY} to forbid a transition.
     * @param logInitial   [K] log-probability initial-state distribution
     * @param transWeight  scalar weighting transition contribution (0 = greedy)
     * @return             length-T array of state indices forming the best path
     */
    public static int[] decode(double[][] emissions, double[][] logTrans,
                                double[] logInitial, double transWeight) {
        int T = emissions.length;
        if (T == 0) return new int[0];
        int K = emissions[0].length;

        double[][] V = new double[T][K];
        int[][] back = new int[T][K];

        for (int k = 0; k < K; k++) {
            V[0][k] = emissions[0][k] + transWeight * logInitial[k];
        }
        for (int t = 1; t < T; t++) {
            for (int k = 0; k < K; k++) {
                int bestPrev = 0;
                double bestScore = V[t - 1][0] + transWeight * logTrans[0][k];
                for (int prev = 1; prev < K; prev++) {
                    double s = V[t - 1][prev] + transWeight * logTrans[prev][k];
                    if (s > bestScore) { bestScore = s; bestPrev = prev; }
                }
                V[t][k] = emissions[t][k] + bestScore;
                back[t][k] = bestPrev;
            }
        }
        int finalBest = 0;
        for (int k = 1; k < K; k++) if (V[T - 1][k] > V[T - 1][finalBest]) finalBest = k;
        int[] path = new int[T];
        path[T - 1] = finalBest;
        for (int t = T - 1; t > 0; t--) path[t - 1] = back[t][path[t]];
        return path;
    }
}
