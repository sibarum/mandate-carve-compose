package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * KV cache foundation, training side: gradients flow into the embedding table
 * and produce structured representations.
 *
 *   Diagnostic 1 (plumbing) — direct supervised: each symbol's embedding is
 *   pulled toward an explicit target vector via EmbedSymbol.backward / step.
 *   Verifies that SGD updates land on the right table row and converge.
 *
 *   Diagnostic 2 (semantic) — contrastive: two groups of symbols, with
 *   pull-within and push-between gradients applied directly through the
 *   table. After training, within-group cosine similarity must exceed
 *   between-group cosine similarity by a clear margin. This is the smallest
 *   demonstration that the cache can carry trained semantic structure, not
 *   just memoized random vectors.
 *
 * Initial embeddings are random (seed-deterministic). Convergence here is the
 * point of the demo, not the magnitudes of the loss values.
 */
public final class EmbeddingTrainingDemo {

    public static void main(String[] args) {
        runDirectSupervised();
        System.out.println();
        runContrastive();
        System.out.println("\nKV cache training: SGD flows through the embedding table; "
                + "trained embeddings carry semantic similarity structure.");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 1: direct supervised training via EmbedSymbol primitive.
    // ---------------------------------------------------------------------

    private static void runDirectSupervised() {
        System.out.println("=== Diagnostic 1: direct supervised training (EmbedSymbol.backward + step) ===");
        int dim = 8;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, 42L);
        EmbedSymbol embed = new EmbedSymbol(table);

        // Distinguishable target vectors per symbol (one-hot in the first dims).
        List<String> syms = List.of("alpha", "beta", "gamma");
        double[][] targets = {
                oneHot(dim, 0),
                oneHot(dim, 1),
                oneHot(dim, 2),
        };

        // Pre-train cosine sim from each embedding to its target — should be near 0.
        for (int i = 0; i < syms.size(); i++) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(syms.get(i))));
            System.out.printf(Locale.ROOT, "  pre  cos(embed(%s), target_%d) = %+.3f%n",
                    syms.get(i), i, cosine(v.data(), targets[i]));
        }

        double lr = 0.5;
        int epochs = 200;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < syms.size(); i++) {
                embed.apply(List.of(new StringValue(syms.get(i))));
                embed.backward(new MatrixValue(targets[i]));
                embed.step(lr);
            }
        }

        // Post-train cosine sim — should be near 1.
        boolean allConverged = true;
        for (int i = 0; i < syms.size(); i++) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(syms.get(i))));
            double cos = cosine(v.data(), targets[i]);
            System.out.printf(Locale.ROOT, "  post cos(embed(%s), target_%d) = %+.3f%n",
                    syms.get(i), i, cos);
            if (cos < 0.99) allConverged = false;
        }
        require(allConverged, "direct training did not converge to within cos > 0.99 of every target");
        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 2: contrastive training over two equivalence groups.
    // ---------------------------------------------------------------------

    private static void runContrastive() {
        System.out.println("=== Diagnostic 2: contrastive training (within-group pull, between-group push) ===");
        int dim = 16;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, 7L);
        List<String> groupA = List.of("a", "b", "c");
        List<String> groupB = List.of("x", "y", "z");

        // Force lazy init for all symbols up front so initial state is the same regardless of training order.
        for (String s : groupA) table.embed(s);
        for (String s : groupB) table.embed(s);

        // Pre-training: average within / between cosine.
        double preWithin = avgWithinCosine(table, groupA, groupB);
        double preBetween = avgBetweenCosine(table, groupA, groupB);
        System.out.printf(Locale.ROOT, "  pre   avg within=%+.3f   avg between=%+.3f%n", preWithin, preBetween);

        double lr = 0.05;
        int epochs = 600;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Pull within group A.
            pullPairs(table, groupA, lr);
            // Pull within group B.
            pullPairs(table, groupB, lr);
            // Push between A and B.
            pushPairs(table, groupA, groupB, lr);
        }

        double postWithin = avgWithinCosine(table, groupA, groupB);
        double postBetween = avgBetweenCosine(table, groupA, groupB);
        double margin = postWithin - postBetween;
        System.out.printf(Locale.ROOT, "  post  avg within=%+.3f   avg between=%+.3f   margin=%+.3f%n",
                postWithin, postBetween, margin);

        require(postWithin > 0.8,
                "within-group cosine did not converge above 0.8: " + postWithin);
        require(margin > 0.5,
                "within-vs-between margin did not exceed 0.5: " + margin);

        // Per-pair table for visibility.
        System.out.println("  per-pair within group A:");
        for (int i = 0; i < groupA.size(); i++) {
            for (int j = i + 1; j < groupA.size(); j++) {
                double c = cosine(table.embed(groupA.get(i)), table.embed(groupA.get(j)));
                System.out.printf(Locale.ROOT, "    cos(%s, %s) = %+.3f%n",
                        groupA.get(i), groupA.get(j), c);
            }
        }
        System.out.println("  per-pair across groups (sample):");
        for (int i = 0; i < groupA.size(); i++) {
            double c = cosine(table.embed(groupA.get(i)), table.embed(groupB.get(i)));
            System.out.printf(Locale.ROOT, "    cos(%s, %s) = %+.3f%n",
                    groupA.get(i), groupB.get(i), c);
        }
        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Contrastive helpers: pull/push gradients applied directly to the table.
    //
    // pull (target = v): gradient = u - v, step moves u toward v.
    // push (target = -v): gradient = u + v, step moves u away from v
    //                     (toward -v in vector space).
    // ---------------------------------------------------------------------

    private static void pullPairs(SymbolEmbeddingTable table, List<String> group, double lr) {
        for (int i = 0; i < group.size(); i++) {
            for (int j = i + 1; j < group.size(); j++) {
                double[] u = table.embed(group.get(i));
                double[] v = table.embed(group.get(j));
                double[] gu = sub(u, v);
                double[] gv = sub(v, u);
                table.update(group.get(i), gu, lr);
                table.update(group.get(j), gv, lr);
            }
        }
    }

    private static void pushPairs(SymbolEmbeddingTable table, List<String> g1, List<String> g2, double lr) {
        for (String s1 : g1) {
            for (String s2 : g2) {
                double[] u = table.embed(s1);
                double[] v = table.embed(s2);
                double[] gu = add(u, v);
                double[] gv = add(v, u);
                table.update(s1, gu, lr);
                table.update(s2, gv, lr);
            }
        }
    }

    private static double avgWithinCosine(SymbolEmbeddingTable table, List<String> a, List<String> b) {
        double sum = 0.0;
        int n = 0;
        for (List<String> g : List.of(a, b)) {
            for (int i = 0; i < g.size(); i++) {
                for (int j = i + 1; j < g.size(); j++) {
                    sum += cosine(table.embed(g.get(i)), table.embed(g.get(j)));
                    n++;
                }
            }
        }
        return sum / n;
    }

    private static double avgBetweenCosine(SymbolEmbeddingTable table, List<String> a, List<String> b) {
        double sum = 0.0;
        int n = 0;
        for (String s1 : a) {
            for (String s2 : b) {
                sum += cosine(table.embed(s1), table.embed(s2));
                n++;
            }
        }
        return sum / n;
    }

    // ---------------------------------------------------------------------
    // Vector utilities.
    // ---------------------------------------------------------------------

    private static double cosine(double[] a, double[] b) {
        double na = 0, nb = 0, d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return d / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private static double[] sub(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
        return r;
    }

    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
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
