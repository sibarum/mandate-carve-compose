package sibarum.elden.demo;

import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;
import sibarum.elden.training.TaggerPipeline;

import java.util.ArrayList;
import java.util.List;

/**
 * Run the trained binary span tagger on Ranni's questline — items that were
 * NEVER seen in training. For each item: tokenize, score every token,
 * aggregate consecutive above-threshold tokens into predicted spans, print.
 *
 * Held-out test in the loosest sense: the items aren't a true random split,
 * but they share zero overlap with the four annotated storylines and contain
 * a mix of (a) entities the trained tagger has seen in other contexts
 * (Marika, Godwyn, Ranni, Two Fingers references) and (b) entities entirely
 * new to it (Nokron, Blaidd, Manus Celes, Fingerslayer Blade, …).
 *
 * The signal we're looking for: does it tag the right things?
 */
public final class RanniInferenceDemo {

    public static void main(String[] args) {
        System.out.println("Training tagger on Shattering + Millicent + DungEater + Volcano Manor…");
        TaggerPipeline pipeline = TaggerPipeline.trainDefault();
        System.out.println("Trained. Running inference on Ranni's questline (unseen).");
        System.out.println();

        List<Item> items = RannisQuestline.items();
        int totalTokens = 0, predictedPositive = 0;
        List<String> predictedSpansAcrossCorpus = new ArrayList<>();

        for (Item item : items) {
            System.out.println("================================================================");
            System.out.println("item: " + item.name());
            System.out.println("================================================================");

            List<OffsetToken> tokens = OffsetTokenizer.tokenize(item.description());
            List<String> texts = tokens.stream().map(OffsetToken::text).toList();
            double[] scores = new double[tokens.size()];
            boolean[] positive = new boolean[tokens.size()];

            for (int i = 0; i < tokens.size(); i++) {
                double[] cv = pipeline.ctx.encode(texts, i);
                scores[i] = pipeline.score(cv);
                positive[i] = scores[i] >= pipeline.threshold;
            }
            totalTokens += tokens.size();
            for (boolean p : positive) if (p) predictedPositive++;

            // Aggregate consecutive positives into spans.
            List<String> itemSpans = new ArrayList<>();
            StringBuilder cur = new StringBuilder();
            for (int i = 0; i < tokens.size(); i++) {
                if (positive[i]) {
                    if (cur.length() > 0) cur.append(' ');
                    cur.append(tokens.get(i).text());
                } else if (cur.length() > 0) {
                    itemSpans.add(cur.toString());
                    cur.setLength(0);
                }
            }
            if (cur.length() > 0) itemSpans.add(cur.toString());
            predictedSpansAcrossCorpus.addAll(itemSpans);

            System.out.println("predicted spans (" + itemSpans.size() + "):");
            for (String s : itemSpans) {
                System.out.println("  • " + s);
            }
            System.out.println();

            // Highest-scoring tokens — useful for diagnosing borderline calls.
            System.out.println("top 10 tokens by score:");
            Integer[] idx = new Integer[tokens.size()];
            for (int i = 0; i < tokens.size(); i++) idx[i] = i;
            java.util.Arrays.sort(idx, (a, b) -> Double.compare(scores[b], scores[a]));
            for (int j = 0; j < Math.min(10, idx.length); j++) {
                int i = idx[j];
                System.out.printf("  %5.2f  %s'%s'%n", scores[i], positive[i] ? "* " : "  ", tokens.get(i).text());
            }
            System.out.println();
        }

        System.out.println("================================================================");
        System.out.println("Corpus-level summary");
        System.out.println("================================================================");
        System.out.println("items inferred:           " + items.size());
        System.out.println("total tokens:             " + totalTokens);
        System.out.println("predicted positive:       " + predictedPositive);
        System.out.printf ("positive rate:            %.1f%%%n", 100.0 * predictedPositive / totalTokens);
        System.out.println("predicted spans total:    " + predictedSpansAcrossCorpus.size());
    }
}
