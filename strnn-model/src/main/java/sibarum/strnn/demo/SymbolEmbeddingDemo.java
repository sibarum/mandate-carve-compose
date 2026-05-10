package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * KV cache foundation: symbol &lt;-&gt; vector.
 *
 * No carver, no mandates — just the embedding substrate. The diagnostics
 * pin down five contracts the first KV cache must obey:
 *
 *   1) Distinct symbols get distinct vectors.
 *   2) Repeated embed of the same symbol returns the same vector (memoized).
 *   3) Reverse lookup recovers the symbol from its exact embedding.
 *   4) Reverse lookup is robust to small perturbations of the query
 *      (cosine-nearest does the right thing).
 *   5) Fresh symbols are added on demand without disturbing existing entries.
 *
 * Initial vectors are random; semantic encoding is the trainer's job and
 * not exercised here. The point of this phase is the substrate.
 */
public final class SymbolEmbeddingDemo {

    public static void main(String[] args) {
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(8, 42L);
        EmbedSymbol embed = new EmbedSymbol(table);
        LookupSymbol lookup = new LookupSymbol(table);

        System.out.println("=== Diagnostic 1: distinct symbols, distinct vectors ===");
        List<String> syms = List.of("x", "y", "z");
        for (String s : syms) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            System.out.printf("  embed(%s) = %s%n", s, format(v.data()));
        }
        for (int i = 0; i < syms.size(); i++) {
            MatrixValue a = (MatrixValue) embed.apply(List.of(new StringValue(syms.get(i))));
            for (int j = i + 1; j < syms.size(); j++) {
                MatrixValue b = (MatrixValue) embed.apply(List.of(new StringValue(syms.get(j))));
                require(!Arrays.equals(a.data(), b.data()),
                        "embeddings collided for " + syms.get(i) + " and " + syms.get(j));
            }
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 2: memoization (same symbol -> same vector) ===");
        MatrixValue first = (MatrixValue) embed.apply(List.of(new StringValue("x")));
        MatrixValue second = (MatrixValue) embed.apply(List.of(new StringValue("x")));
        require(Arrays.equals(first.data(), second.data()),
                "embed('x') returned different vectors on repeated calls");
        System.out.println("  embed('x') consistent across calls");
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 3: exact reverse lookup ===");
        for (String s : syms) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            StringValue back = (StringValue) lookup.apply(List.of(v));
            System.out.printf("  embed(%s) -> %s -> '%s'%n", s, format(v.data()), back.s());
            require(back.s().equals(s), "reverse lookup failed: " + s + " -> " + back.s());
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 4: nearest-neighbour robustness to perturbation ===");
        Random pertRng = new Random(1234L);
        double noise = 0.05;
        for (String s : syms) {
            MatrixValue v = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            double[] perturbed = v.data().clone();
            for (int i = 0; i < perturbed.length; i++) {
                perturbed[i] += (pertRng.nextDouble() * 2.0 - 1.0) * noise;
            }
            StringValue back = (StringValue) lookup.apply(List.of(new MatrixValue(perturbed)));
            require(back.s().equals(s),
                    "perturbed query for '" + s + "' resolved to '" + back.s() + "'");
            System.out.printf("  perturbed(%s, noise=%.2f) -> '%s'%n", s, noise, back.s());
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 5: lazy growth on unseen symbols ===");
        int sizeBefore = table.size();
        embed.apply(List.of(new StringValue("foo")));
        require(table.size() == sizeBefore + 1, "table did not grow by 1 on new symbol");
        for (String s : syms) {
            MatrixValue same = (MatrixValue) embed.apply(List.of(new StringValue(s)));
            StringValue back = (StringValue) lookup.apply(List.of(same));
            require(back.s().equals(s),
                    "after growth, '" + s + "' no longer resolves correctly: " + back.s());
        }
        System.out.printf("  added 'foo'; table size %d -> %d; old symbols intact%n",
                sizeBefore, table.size());
        System.out.println("  PASS");

        System.out.println("\nKV cache foundation: symbol <-> vector mapping over "
                + table.size() + " symbols at dim " + table.dim() + ".");
    }

    private static String format(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format(Locale.ROOT, "%+.3f", v[i]));
        }
        return sb.append("]").toString();
    }

    private static void require(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }
}
