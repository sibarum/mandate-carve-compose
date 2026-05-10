package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Proof of concept for the symbol &lt;-&gt; vector substrate as a content-addressable
 * memory. Composes the pieces — table, EmbedSymbol, LookupSymbol, training —
 * and demonstrates that the trained substrate actually does the job claimed:
 *
 *   1) N symbols are trained to mutually distinguishable target positions
 *      (orthogonal one-hot directions). After training, every symbol's
 *      embedding round-trips through the lookup primitive correctly.
 *
 *   2) Recall is robust under additive Gaussian noise on the query vector,
 *      degrading gracefully rather than collapsing. We sweep noise levels
 *      and report recall accuracy at each — a recall curve, not a single
 *      pass/fail bit.
 *
 *   3) The concept earns its name only if recall is high at low noise AND
 *      degrades smoothly. Random chance for N=20 symbols would be 5%
 *      recall; we want the cache to do meaningfully better than that across
 *      a useful noise range.
 */
public final class TrainedRecallDemo {

    public static void main(String[] args) {
        int n = 20;
        int dim = 32;
        long seed = 2024L;

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, seed);
        EmbedSymbol embed = new EmbedSymbol(table);
        LookupSymbol lookup = new LookupSymbol(table);

        List<String> syms = new ArrayList<>(n);
        for (int i = 0; i < n; i++) syms.add("s" + i);

        // ---- train: each symbol toward a distinct one-hot target in dim space ----
        // dim >= n means every symbol gets its own basis direction; the targets are
        // mutually orthogonal so the recall problem is well-posed.
        require(dim >= n, "dim must be >= n for orthogonal targets in this demo");
        double[][] targets = new double[n][];
        for (int i = 0; i < n; i++) targets[i] = oneHot(dim, i);

        double lr = 0.3;
        int epochs = 300;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                embed.apply(List.of(new StringValue(syms.get(i))));
                embed.backward(new MatrixValue(targets[i]));
                embed.step(lr);
            }
        }

        // ---- diagnostic 1: clean round-trip on every trained symbol ----
        System.out.println("=== Diagnostic 1: clean round-trip on " + n + " symbols ===");
        int cleanHits = 0;
        for (String s : syms) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            StringValue back = (StringValue) lookup.apply(List.of(v));
            if (back.s().equals(s)) cleanHits++;
        }
        System.out.printf(Locale.ROOT, "  recall = %d / %d (%.1f%%)%n",
                cleanHits, n, 100.0 * cleanHits / n);
        require(cleanHits == n, "clean round-trip failed for " + (n - cleanHits) + " symbols");
        System.out.println("  PASS");

        // ---- diagnostic 2: recall under increasing additive noise ----
        // Noise is Gaussian, magnitude expressed as a multiple of the embedding's
        // L2 norm. At noise=0 we expect 100%; at noise → infinity, ~5% (chance).
        System.out.println("\n=== Diagnostic 2: recall under additive noise (queries per symbol = 20) ===");
        double[] noiseLevels = {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0};
        int queriesPerSymbol = 20;
        double chance = 1.0 / n;
        Random noiseRng = new Random(0xC0FFEEL);

        System.out.printf(Locale.ROOT, "  chance baseline = %.1f%%%n", 100.0 * chance);
        for (double noise : noiseLevels) {
            int hits = 0;
            int total = 0;
            for (String s : syms) {
                MatrixValue clean = (MatrixValue) embed.apply(List.of(new StringValue(s)));
                double cleanNorm = norm(clean.data());
                for (int q = 0; q < queriesPerSymbol; q++) {
                    double[] noisy = clean.data().clone();
                    for (int k = 0; k < noisy.length; k++) {
                        noisy[k] += noiseRng.nextGaussian() * noise * cleanNorm / Math.sqrt(noisy.length);
                    }
                    StringValue back = (StringValue) lookup.apply(List.of(new MatrixValue(noisy)));
                    if (back.s().equals(s)) hits++;
                    total++;
                }
            }
            double accuracy = (double) hits / total;
            System.out.printf(Locale.ROOT, "  noise = %.2f * |emb|   recall = %4d / %4d (%5.1f%%)%n",
                    noise, hits, total, 100.0 * accuracy);
        }

        // ---- diagnostic 3: the concept earns its name ----
        // Two structural checks: clean recall is perfect, AND moderate-noise recall
        // is well above chance. If either fails, the substrate isn't ready for
        // composition into anything bigger.
        System.out.println("\n=== Diagnostic 3: concept earns its name ===");
        double moderateNoiseAccuracy = noiseRecall(embed, lookup, syms, 0.5, queriesPerSymbol, new Random(7L));
        System.out.printf(Locale.ROOT, "  recall @ noise=0.5*|emb|  = %.1f%%%n", 100.0 * moderateNoiseAccuracy);
        System.out.printf(Locale.ROOT, "  chance baseline           = %.1f%%%n", 100.0 * chance);
        require(moderateNoiseAccuracy > 0.9,
                "moderate-noise recall did not exceed 90% (got " + (100.0 * moderateNoiseAccuracy) + "%)");
        require(moderateNoiseAccuracy > chance * 10,
                "moderate-noise recall not meaningfully above chance");
        System.out.println("  PASS — substrate works end-to-end as a content-addressable memory.");

        System.out.println("\nProof of concept: " + n + " symbols, dim " + dim
                + ", trained substrate recovers the right key under realistic noise.");
    }

    private static double noiseRecall(EmbedSymbol embed, LookupSymbol lookup, List<String> syms,
                                      double noise, int queriesPerSymbol, Random rng) {
        int hits = 0;
        int total = 0;
        for (String s : syms) {
            MatrixValue clean = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            double cleanNorm = norm(clean.data());
            for (int q = 0; q < queriesPerSymbol; q++) {
                double[] noisy = clean.data().clone();
                for (int k = 0; k < noisy.length; k++) {
                    noisy[k] += rng.nextGaussian() * noise * cleanNorm / Math.sqrt(noisy.length);
                }
                StringValue back = (StringValue) lookup.apply(List.of(new MatrixValue(noisy)));
                if (back.s().equals(s)) hits++;
                total++;
            }
        }
        return (double) hits / total;
    }

    private static double norm(double[] a) {
        double s = 0;
        for (double v : a) s += v * v;
        return Math.sqrt(s);
    }

    private static double[] oneHot(int dim, int idx) {
        double[] v = new double[dim];
        v[idx] = 1.0;
        return v;
    }

    private static void require(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }
}
