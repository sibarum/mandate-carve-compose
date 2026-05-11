package sibarum.elden.data;

import sibarum.elden.data.LexicanumParser.Entry;
import sibarum.elden.data.ParquetBioLoader.Sentence;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * For each Lexicanum entry, bootstrap BIO labels by finding mentions of the
 * entry's title and aliases in its own prose. Each match becomes B-ENT (first
 * token) + I-ENT (continuation tokens).
 *
 * Surface-form matching:
 *   - Both the entry's surface forms and the prose are tokenized with the
 *     same {@link OffsetTokenizer}, so token boundaries line up.
 *   - Comparison is case-insensitive (lore prose often varies capitalization).
 *   - Longest surface forms are matched first so multi-word names take
 *     precedence over their substrings ("Black Knife Assassin" beats
 *     "Knife").
 *   - Already-labeled positions are not overwritten — the first (longest)
 *     match wins.
 *
 * Output is a list of pre-tokenized BIO-labeled sentences in the same format
 * as {@link ParquetBioLoader.Sentence}, so the training pipeline can consume
 * them with no adapter.
 *
 * Honest limitation: this only labels self-mentions (the entry's own title
 * within its own prose). A later iteration could cross-reference titles
 * across the whole corpus — every Lexicanum entry could supply gold-standard
 * mentions of every OTHER entry's title within its prose, dramatically
 * expanding labeled examples. Out of scope for v0.
 */
public final class LexicanumBioLabeler {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;

    private LexicanumBioLabeler() {}

    public static List<Sentence> label(List<Entry> entries) {
        List<Sentence> out = new ArrayList<>();
        for (Entry e : entries) {
            if (e.prose() == null || e.prose().isBlank()) continue;

            List<OffsetToken> proseToks = OffsetTokenizer.tokenize(e.prose());
            int n = proseToks.size();
            if (n == 0) continue;
            int[] bio = new int[n];

            // Collect surface forms (title + aliases), tokenized for matching.
            List<List<String>> surfaces = new ArrayList<>();
            addSurface(surfaces, e.title());
            for (String alias : e.aliases()) addSurface(surfaces, alias);

            // Longest first so multi-word names beat single tokens.
            surfaces.sort(Comparator.comparingInt((List<String> s) -> s.size()).reversed());

            for (List<String> surface : surfaces) {
                int sLen = surface.size();
                if (sLen == 0) continue;
                for (int i = 0; i + sLen <= n; i++) {
                    boolean overlapsExisting = false;
                    for (int j = 0; j < sLen; j++) {
                        if (bio[i + j] != LABEL_O) { overlapsExisting = true; break; }
                    }
                    if (overlapsExisting) continue;

                    boolean match = true;
                    for (int j = 0; j < sLen; j++) {
                        if (!proseToks.get(i + j).text().equalsIgnoreCase(surface.get(j))) {
                            match = false; break;
                        }
                    }
                    if (!match) continue;

                    bio[i] = LABEL_B;
                    for (int j = 1; j < sLen; j++) bio[i + j] = LABEL_I;
                }
            }

            List<String> forms = new ArrayList<>(n);
            for (OffsetToken t : proseToks) forms.add(t.text());
            out.add(new Sentence(forms, bio));
        }
        return out;
    }

    private static void addSurface(List<List<String>> surfaces, String text) {
        if (text == null || text.isBlank()) return;
        List<String> toks = new ArrayList<>();
        for (OffsetToken t : OffsetTokenizer.tokenize(text)) toks.add(t.text());
        if (!toks.isEmpty()) surfaces.add(toks);
    }
}
