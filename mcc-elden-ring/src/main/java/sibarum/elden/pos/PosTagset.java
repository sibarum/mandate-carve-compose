package sibarum.elden.pos;

import java.util.List;

/**
 * Universal POS tagset (Universal Dependencies v2). 17 coarse tags covering
 * the major grammatical roles. Order is fixed so the {@code int} class index
 * maps consistently to a one-hot vector and back.
 *
 * Reference: https://universaldependencies.org/u/pos/
 */
public final class PosTagset {

    public static final List<String> UPOS = List.of(
            "ADJ",    //  0  adjective
            "ADP",    //  1  adposition  (preposition / postposition)
            "ADV",    //  2  adverb
            "AUX",    //  3  auxiliary verb
            "CCONJ",  //  4  coordinating conjunction
            "DET",    //  5  determiner
            "INTJ",   //  6  interjection
            "NOUN",   //  7  common noun
            "NUM",    //  8  numeral
            "PART",   //  9  particle
            "PRON",   // 10  pronoun
            "PROPN",  // 11  proper noun
            "PUNCT",  // 12  punctuation
            "SCONJ",  // 13  subordinating conjunction
            "SYM",    // 14  symbol
            "VERB",   // 15  verb
            "X"       // 16  other / unknown
    );

    private PosTagset() {}

    public static int indexOf(String tag) {
        return UPOS.indexOf(tag);
    }

    public static String tagAt(int idx) {
        return UPOS.get(idx);
    }

    public static int size() {
        return UPOS.size();
    }
}
