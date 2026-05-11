package sibarum.elden.training;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.EntitySpan;
import sibarum.elden.embedding.ContextEncoder;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts annotated items into per-token training examples for the binary
 * span tagger. Each example is (context vector, label) where the label is
 * 1.0 if the token sits entirely within any EntitySpan and 0.0 otherwise.
 *
 * Alignment rule: a token at character range [tokenStart, tokenEnd) is
 * "in" an entity span [spanStart, spanEnd) iff
 * tokenStart >= spanStart and tokenEnd <= spanEnd. This handles the common
 * case where entities align to whole tokens; tokens that overlap a span
 * boundary are treated as out-of-entity (a noise source we accept for v0).
 */
public final class TaggingTrainingData {

    public record Example(double[] contextVector, double label, String token) {}

    private TaggingTrainingData() {}

    public static List<Example> extract(AnnotatedItem item, ContextEncoder ctx) {
        List<OffsetToken> tokens = OffsetTokenizer.tokenize(item.rawText());
        boolean[] inEntity = new boolean[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            OffsetToken t = tokens.get(i);
            for (EntitySpan span : item.spans()) {
                if (t.startChar() >= span.start() && t.endChar() <= span.end()) {
                    inEntity[i] = true;
                    break;
                }
            }
        }
        List<String> texts = tokens.stream().map(OffsetToken::text).toList();
        List<Example> out = new ArrayList<>(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            double[] cv = ctx.encode(texts, i);
            out.add(new Example(cv, inEntity[i] ? 1.0 : 0.0, texts.get(i)));
        }
        return out;
    }

    public static List<Example> extractAll(List<AnnotatedItem> items, ContextEncoder ctx) {
        List<Example> all = new ArrayList<>();
        for (AnnotatedItem item : items) {
            all.addAll(extract(item, ctx));
        }
        return all;
    }
}
