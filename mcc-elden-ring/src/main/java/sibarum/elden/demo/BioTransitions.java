package sibarum.elden.demo;

/**
 * Hand-set transition log-probabilities for BIO entity-span decoding.
 *
 * The 3-label set is {O=0, B=1, I=2}. Unlike the POS bigram transitions
 * (which were learned from data and turned out to have no resolving power
 * over a strong contextual emission model — iter 14 null result), these are
 * <b>hand-set</b> structural constraints:
 *
 * <ul>
 *   <li><b>Structurally impossible</b> (large negative cost): {@code O -> I}
 *       cannot happen because I must follow B or I in BIO; I at sentence
 *       start is also impossible.</li>
 *   <li><b>Soft preference</b>: {@code I -> B} carries a moderate penalty.
 *       This is the iter-16 targeted fix — by iter 15 the BIO tagger still
 *       emits B at every capitalized token even when the surrounding context
 *       is in-span, splitting "Remembrance of the Baleful Shadow" at "the |
 *       Baleful". The penalty makes the decoder prefer I-continuation over
 *       starting a new entity when the previous token was I.</li>
 *   <li><b>Soft preference</b>: {@code B -> B} carries a small penalty.
 *       Adjacent single-token spans are rare in real prose; this discourages
 *       the model from emitting `B B B` sequences when `B I I` is structurally
 *       cleaner.</li>
 * </ul>
 *
 * Use {@link sibarum.elden.pos.ViterbiDecoder#decode(double[][], double[][],
 * double[], double)} with the emission scores from the BIO classifier head
 * and {@code transWeight = 1.0} so the hand-set values take effect at their
 * stated magnitudes.
 */
public final class BioTransitions {

    public static final int LABEL_O = 0;
    public static final int LABEL_B = 1;
    public static final int LABEL_I = 2;
    public static final int NUM_LABELS = 3;

    /**
     * Cost used for transitions we treat as structurally impossible. Large
     * enough relative to BIO emission scores (~[-0.5, 1.5]) to be unreachable
     * at {@code transWeight = 1.0}, finite so it doesn't NaN at weight 0.
     */
    public static final double FORBIDDEN = -50.0;

    private BioTransitions() {}

    /** A fresh 3x3 transition matrix {@code logTrans[prev][next]}. */
    public static double[][] logTrans() {
        double[][] t = new double[NUM_LABELS][NUM_LABELS];
        // O -> ?
        t[LABEL_O][LABEL_O] = 0.0;
        t[LABEL_O][LABEL_B] = 0.0;
        t[LABEL_O][LABEL_I] = FORBIDDEN;  // structural
        // B -> ?
        t[LABEL_B][LABEL_O] = 0.0;
        t[LABEL_B][LABEL_B] = -1.0;       // adjacent single-token spans rare
        t[LABEL_B][LABEL_I] = 0.0;
        // I -> ?
        t[LABEL_I][LABEL_O] = 0.0;
        t[LABEL_I][LABEL_B] = -2.0;       // the iter-16 targeted penalty
        t[LABEL_I][LABEL_I] = 0.0;
        return t;
    }

    /** A fresh length-3 initial-state log-probability vector. */
    public static double[] logInitial() {
        double[] li = new double[NUM_LABELS];
        li[LABEL_O] = Math.log(0.5);
        li[LABEL_B] = Math.log(0.5);
        li[LABEL_I] = FORBIDDEN;  // sentence cannot start with I
        return li;
    }
}
