package sibarum.elden.embedding;

import sibarum.elden.corpus.DungEaterQuestline;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.MillicentQuestline;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.corpus.ShatteringEra;
import sibarum.elden.corpus.VolcanoManor;
import sibarum.strnn.primitive.TextTokenize;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Builds the token vocabulary over the full Elden Ring corpus by tokenizing
 * every item description and collecting unique tokens. Insertion order is
 * preserved so the vocabulary is deterministic for a given corpus.
 */
public final class CorpusVocabulary {

    private CorpusVocabulary() {}

    public static List<Item> allItems() {
        List<Item> all = new ArrayList<>();
        all.addAll(ShatteringEra.items());
        all.addAll(RannisQuestline.items());
        all.addAll(VolcanoManor.items());
        all.addAll(DungEaterQuestline.items());
        all.addAll(MillicentQuestline.items());
        return all;
    }

    public static Set<String> tokens() {
        TextTokenize tokenizer = new TextTokenize();
        Set<String> vocab = new LinkedHashSet<>();
        for (Item item : allItems()) {
            TokenListValue out = (TokenListValue) tokenizer.apply(List.of(new StringValue(item.description())));
            vocab.addAll(out.tokens());
        }
        return vocab;
    }
}
