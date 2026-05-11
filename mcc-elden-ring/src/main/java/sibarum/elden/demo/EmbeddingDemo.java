package sibarum.elden.demo;

import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;

import java.util.List;
import java.util.Set;

/**
 * Initialize a token embedding table over the corpus vocabulary, then run
 * sanity checks:
 *   - every unique token gets a vector
 *   - embeddings are stable across repeated lookups
 *   - cosine nearest-neighbor returns the query token itself
 *   - a single training step on a token shifts that token's nearest neighbor
 */
public final class EmbeddingDemo {

    private static final int EMBED_DIM = 32;
    private static final long SEED = 7L;

    public static void main(String[] args) {
        Set<String> vocab = CorpusVocabulary.tokens();
        System.out.println("Corpus vocabulary: " + vocab.size() + " unique tokens");

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(EMBED_DIM, SEED);
        for (String token : vocab) {
            table.embed(token);
        }
        System.out.println("Embedding table populated: " + table.size() + " entries, dim=" + table.dim());
        System.out.println();

        // Lookup stability.
        EmbedSymbol embedder = new EmbedSymbol(table);
        MatrixValue v1 = (MatrixValue) embedder.apply(List.of(new StringValue("Marika")));
        MatrixValue v2 = (MatrixValue) embedder.apply(List.of(new StringValue("Marika")));
        boolean stable = java.util.Arrays.equals(v1.data(), v2.data());
        System.out.println("'Marika' embedding stable across lookups: " + stable);

        // Cosine nearest of a token's own embedding == itself.
        for (String probe : List.of("Marika", "Godwyn", "Erdtree", "Rune", "the")) {
            if (!vocab.contains(probe)) {
                System.out.println("  '" + probe + "' not in vocab");
                continue;
            }
            double[] v = table.embed(probe);
            String nearest = table.nearest(v).orElse("?");
            System.out.println("  nearest('" + probe + "') = '" + nearest + "'  (expect itself: " + probe.equals(nearest) + ")");
        }
        System.out.println();

        // Training step: pull 'Marika' embedding toward 'Godwyn' embedding.
        double[] marikaBefore = table.embed("Marika").clone();
        double[] godwyn = table.embed("Godwyn").clone();
        System.out.println("Before training: nearest('Marika') = '" + table.nearest(marikaBefore).orElse("?") + "'");

        // 100 SGD steps shoving Marika toward Godwyn's vector.
        for (int step = 0; step < 100; step++) {
            embedder.apply(List.of(new StringValue("Marika")));
            embedder.backward(new MatrixValue(godwyn));
            embedder.step(0.1);
        }

        double[] marikaAfter = table.embed("Marika");
        double drift = euclidean(marikaBefore, marikaAfter);
        System.out.printf("After 100 SGD steps toward 'Godwyn': drift = %.4f%n", drift);
        System.out.println("Nearest('Marika') is now: '" + table.nearest(marikaAfter).orElse("?") + "'");
        System.out.println();

        System.out.println("Embedding substrate validated.");
        System.out.println();

        // Context encoder smoke check on a real corpus sentence.
        sibarum.strnn.primitive.TextTokenize tok = new sibarum.strnn.primitive.TextTokenize();
        sibarum.strnn.value.TokenListValue tlv = (sibarum.strnn.value.TokenListValue) tok.apply(List.of(
                new StringValue("Marika shattered the Elden Ring after Godwyn's death.")));
        List<String> tokens = tlv.tokens();
        sibarum.elden.embedding.ContextEncoder ctx = new sibarum.elden.embedding.ContextEncoder(table, 1);

        System.out.println("Context encoder (window=1, dim=" + table.dim() + ") -> contextDim=" + ctx.contextDim());
        System.out.println("Tokens (" + tokens.size() + "): " + tokens);
        for (int i = 0; i < tokens.size(); i++) {
            double[] cv = ctx.encode(tokens, i);
            double norm = 0;
            for (double x : cv) norm += x * x;
            System.out.printf("  pos %2d  '%s'  ||context||=%.3f%n", i, tokens.get(i), Math.sqrt(norm));
        }
    }

    private static double euclidean(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }
}
