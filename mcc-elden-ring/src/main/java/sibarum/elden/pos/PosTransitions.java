package sibarum.elden.pos;

import sibarum.elden.pos.ConlluParser.Sentence;
import sibarum.elden.pos.ConlluParser.TaggedToken;

import java.util.List;

/**
 * Tag-bigram transition log-probabilities for Viterbi decoding over POS tags.
 * Counts {@code count[prev][next]} from the training sentences, applies
 * Laplace (add-one) smoothing, then converts to log-probability so the decoder
 * can sum scores additively.
 *
 * Stores a separate {@code logInitial[tag]} for the distribution at sentence
 * start — the start-of-sentence prior is structurally distinct from any
 * mid-sentence prev tag (function-word starts are rare; PROPN/PRON/DET
 * dominate).
 */
public final class PosTransitions {

    public final double[][] logTrans;   // [prev][next]
    public final double[] logInitial;   // [tag]

    private PosTransitions(double[][] logTrans, double[] logInitial) {
        this.logTrans = logTrans;
        this.logInitial = logInitial;
    }

    /** Compute transitions from a list of tagged sentences. */
    public static PosTransitions fromSentences(List<Sentence> sentences) {
        int n = PosTagset.size();
        int[][] counts = new int[n][n];
        int[] initialCounts = new int[n];
        int[] rowTotals = new int[n];
        int initialTotal = 0;

        for (Sentence s : sentences) {
            List<TaggedToken> toks = s.tokens();
            if (toks.isEmpty()) continue;
            int prev = PosTagset.indexOf(toks.get(0).upos());
            if (prev >= 0) { initialCounts[prev]++; initialTotal++; }
            for (int i = 1; i < toks.size(); i++) {
                int cur = PosTagset.indexOf(toks.get(i).upos());
                if (prev >= 0 && cur >= 0) {
                    counts[prev][cur]++;
                    rowTotals[prev]++;
                }
                if (cur >= 0) prev = cur;
            }
        }

        double[][] lt = new double[n][n];
        for (int i = 0; i < n; i++) {
            double denom = rowTotals[i] + n;
            for (int j = 0; j < n; j++) {
                lt[i][j] = Math.log((counts[i][j] + 1.0) / denom);
            }
        }
        double[] li = new double[n];
        double initDenom = initialTotal + n;
        for (int j = 0; j < n; j++) {
            li[j] = Math.log((initialCounts[j] + 1.0) / initDenom);
        }
        return new PosTransitions(lt, li);
    }
}
