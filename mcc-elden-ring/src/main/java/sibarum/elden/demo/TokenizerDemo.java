package sibarum.elden.demo;

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
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public final class TokenizerDemo {

    public static void main(String[] args) {
        List<Item> all = new ArrayList<>();
        all.addAll(ShatteringEra.items());
        all.addAll(RannisQuestline.items());
        all.addAll(VolcanoManor.items());
        all.addAll(DungEaterQuestline.items());
        all.addAll(MillicentQuestline.items());

        TextTokenize tokenizer = new TextTokenize();

        int totalTokens = 0;
        Map<String, Integer> vocab = new TreeMap<>();

        System.out.println("Tokenizer dry run across full corpus");
        System.out.println("====================================");

        for (Item item : all) {
            TokenListValue out = (TokenListValue) tokenizer.apply(List.of(new StringValue(item.description())));
            List<String> tokens = out.tokens();
            totalTokens += tokens.size();
            for (String t : tokens) vocab.merge(t, 1, Integer::sum);
        }

        System.out.println("items processed:    " + all.size());
        System.out.println("total tokens:       " + totalTokens);
        System.out.println("unique tokens:      " + vocab.size());
        System.out.printf("avg tokens / item:  %.1f%n", (double) totalTokens / all.size());
        System.out.println();

        System.out.println("first item, first 25 tokens:");
        Item first = all.getFirst();
        TokenListValue firstTokens = (TokenListValue) tokenizer.apply(List.of(new StringValue(first.description())));
        System.out.println("  item: " + first.name());
        StringBuilder line = new StringBuilder("  ");
        for (int i = 0; i < Math.min(25, firstTokens.tokens().size()); i++) {
            line.append('"').append(firstTokens.tokens().get(i)).append("\" ");
        }
        System.out.println(line);
        System.out.println();

        System.out.println("top 20 most-frequent tokens:");
        vocab.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(20)
                .forEach(e -> System.out.printf("  %-20s %d%n", '"' + e.getKey() + '"', e.getValue()));
    }
}
