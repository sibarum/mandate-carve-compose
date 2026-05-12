package sibarum.elden.data;

import sibarum.elden.data.ParquetBioLoader.Sentence;

/**
 * 7-class label space for iter-18 corpus-gradient BIO training.
 *
 * The three corpora supply BIO supervision at three levels of domain
 * specificity, captured by separate class pairs:
 *
 * <pre>
 *   O                                — function words, verbs (cross-corpus)
 *   B-normal,    I-normal             — generic English entities (NLP-KnowledgeGraph)
 *   B-fantasy,   I-fantasy            — fantasy-genre entities (Lexicanum)
 *   B-eldenring, I-eldenring          — Elden Ring entities (JSONL + hand-annotations)
 * </pre>
 *
 * At inference any of the three (B-* + I-*) runs is a positive entity span;
 * the class label tells downstream how confident the model is in domain
 * specificity. Collapse to 3-class BIO when interoperating with existing
 * span-decode utilities.
 */
public final class EntityClasses {

    public static final int O = 0;
    public static final int B_NORMAL = 1;
    public static final int I_NORMAL = 2;
    public static final int B_FANTASY = 3;
    public static final int I_FANTASY = 4;
    public static final int B_ELDENRING = 5;
    public static final int I_ELDENRING = 6;
    public static final int NUM_CLASSES = 7;

    private EntityClasses() {}

    /**
     * Take a 3-class {@link Sentence} (labels {0,1,2}) and produce a 7-class
     * version where each 1 becomes {@code bClass} and each 2 becomes
     * {@code iClass}. O stays O.
     */
    public static Sentence remap(Sentence src, int bClass, int iClass) {
        int[] in = src.bioLabels();
        int[] out = new int[in.length];
        for (int i = 0; i < in.length; i++) {
            if (in[i] == 1) out[i] = bClass;
            else if (in[i] == 2) out[i] = iClass;
            else out[i] = O;
        }
        return new Sentence(src.tokens(), out);
    }

    /** Collapse a 7-class label to {O=0, B=1, I=2}. */
    public static int toBio3(int cls) {
        if (cls == O) return 0;
        if (cls == B_NORMAL || cls == B_FANTASY || cls == B_ELDENRING) return 1;
        return 2;
    }

    /** Three-letter display name for a class. */
    public static String shortName(int cls) {
        return switch (cls) {
            case O -> "O";
            case B_NORMAL -> "B-N"; case I_NORMAL -> "I-N";
            case B_FANTASY -> "B-F"; case I_FANTASY -> "I-F";
            case B_ELDENRING -> "B-E"; case I_ELDENRING -> "I-E";
            default -> "?";
        };
    }

    /** Domain of an entity-class label (B-* or I-*); "O" or "?" otherwise. */
    public static String domain(int cls) {
        return switch (cls) {
            case B_NORMAL, I_NORMAL -> "normal";
            case B_FANTASY, I_FANTASY -> "fantasy";
            case B_ELDENRING, I_ELDENRING -> "eldenring";
            case O -> "O";
            default -> "?";
        };
    }
}
