package sibarum.elden.data;

import sibarum.elden.data.LexicanumParser.Entry;
import sibarum.elden.data.ParquetBioLoader.Sentence;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;

import java.util.ArrayList;
import java.util.List;

/**
 * Cross-entry BIO labeler. Builds a single dictionary of every entry's title
 * + aliases (a global "gazetteer") and then labels every entry's prose
 * against that whole dictionary — not just against the entry's own surface
 * forms.
 *
 * This addresses the precision-recall problem with the self-mention labeler:
 * lore prose mentions dozens of entities by name, and the self-mention
 * labeler tagged most of them as O (false negatives) because they weren't
 * the entry's own title. Cross-entry labeling lights up all those mentions
 * with B-ENT / I-ENT and dramatically raises the positive density.
 *
 * After labeling, an optional density filter keeps only entries whose prose
 * has high entity density (configurable) — useful for building a focused
 * pretraining set where most tokens are entity-relevant rather than diluted
 * by descriptive filler.
 */
public final class LexicanumCrossEntryLabeler {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;

    public record LabelingStats(int totalEntries,
                                 int totalSurfaceForms,
                                 int labeledSentences,
                                 long totalTokens,
                                 long entityTokens,
                                 int sentencesAfterFilter,
                                 long entityTokensAfterFilter) {}

    private LexicanumCrossEntryLabeler() {}

    public static class Result {
        public final List<Sentence> sentences;
        public final LabelingStats stats;
        Result(List<Sentence> sentences, LabelingStats stats) {
            this.sentences = sentences;
            this.stats = stats;
        }
    }

    /**
     * @param minDensity         minimum (entity tokens / total tokens) ratio
     *                           for a sentence to be kept; 0 disables
     * @param minEntityTokens    minimum entity tokens per sentence; 0 disables
     */
    public static Result label(List<Entry> entries, double minDensity, int minEntityTokens) {
        SurfaceTrie trie = new SurfaceTrie();
        int totalSurfaces = 0;
        for (Entry e : entries) {
            if (insertSurface(trie, e.title())) totalSurfaces++;
            for (String alias : e.aliases()) {
                if (insertSurface(trie, alias)) totalSurfaces++;
            }
        }

        List<Sentence> all = new ArrayList<>();
        long totalTokens = 0;
        long entityTokens = 0;

        for (Entry e : entries) {
            if (e.prose() == null || e.prose().isBlank()) continue;
            List<OffsetToken> toks = OffsetTokenizer.tokenize(e.prose());
            if (toks.isEmpty()) continue;
            List<String> forms = new ArrayList<>(toks.size());
            for (OffsetToken t : toks) forms.add(t.text());

            int[] bio = new int[forms.size()];
            int i = 0;
            while (i < forms.size()) {
                int match = trie.longestMatch(forms, i);
                if (match > 0) {
                    bio[i] = LABEL_B;
                    for (int j = 1; j < match; j++) bio[i + j] = LABEL_I;
                    i += match;
                } else {
                    i++;
                }
            }

            int ec = entityCount(bio);
            totalTokens += forms.size();
            entityTokens += ec;
            all.add(new Sentence(forms, bio));
        }

        // Filter by density.
        List<Sentence> kept = new ArrayList<>();
        long keptEntityTokens = 0;
        for (Sentence s : all) {
            int ec = entityCount(s.bioLabels());
            double density = (double) ec / s.tokens().size();
            if (ec < minEntityTokens) continue;
            if (density < minDensity) continue;
            kept.add(s);
            keptEntityTokens += ec;
        }

        LabelingStats stats = new LabelingStats(
                entries.size(), totalSurfaces, all.size(),
                totalTokens, entityTokens,
                kept.size(), keptEntityTokens);
        return new Result(kept, stats);
    }

    private static boolean insertSurface(SurfaceTrie trie, String text) {
        if (text == null || text.isBlank()) return false;
        List<String> toks = new ArrayList<>();
        for (OffsetToken t : OffsetTokenizer.tokenize(text)) toks.add(t.text());
        if (toks.isEmpty()) return false;
        trie.insert(toks);
        return true;
    }

    private static int entityCount(int[] bio) {
        int c = 0;
        for (int b : bio) if (b != LABEL_O) c++;
        return c;
    }
}
