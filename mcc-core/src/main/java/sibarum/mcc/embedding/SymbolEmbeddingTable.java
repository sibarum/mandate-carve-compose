package sibarum.mcc.embedding;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

/**
 * Content-addressable mapping from arbitrary string symbols to
 * fixed-dimensional vectors. Symbols get random vectors on first
 * sight (lazy init); subsequent {@link #embed} calls return the same
 * vector. Reverse lookup ({@link #nearest}) finds the symbol whose
 * stored vector has the highest cosine similarity to a query.
 *
 * <p>Initial vectors are drawn uniformly from
 * {@code [-sqrt(3/dim), +sqrt(3/dim)]} (Xavier-style bound;
 * unit-variance per coordinate). They are meaningless at init —
 * semantic encoding is the trainer's job via {@link #update}.
 *
 * <p>Insertion order is preserved (LinkedHashMap) so {@code nearest}
 * ties resolve deterministically given a deterministic insertion
 * sequence.
 */
public final class SymbolEmbeddingTable {
    private final int dim;
    private final Random rng;
    private final double initBound;
    private final Map<String, double[]> embeddings = new LinkedHashMap<>();

    public SymbolEmbeddingTable(int dim, long seed) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive: " + dim);
        }
        this.dim = dim;
        this.rng = new Random(seed);
        this.initBound = Math.sqrt(3.0 / dim);
    }

    public int dim() { return dim; }
    public int size() { return embeddings.size(); }

    public Set<String> symbols() {
        return Collections.unmodifiableSet(embeddings.keySet());
    }

    /**
     * Direct accessor used by serialization. Returns the table's
     * internal storage for {@code symbol}, or {@code null} if absent.
     * Callers must not mutate the returned array except for in-place
     * gradient updates via {@link #update}.
     */
    public double[] rawVector(String symbol) {
        return embeddings.get(symbol);
    }

    /**
     * Overwrite the embedding for {@code symbol}, allocating it if
     * absent. Used by importers restoring serialized state.
     */
    public void put(String symbol, double[] vector) {
        if (vector.length != dim) {
            throw new IllegalArgumentException(
                    "vector dim " + vector.length + " != table dim " + dim);
        }
        embeddings.put(symbol, vector.clone());
    }

    /**
     * Returns the embedding vector for {@code symbol}, generating a
     * random initial vector if the symbol has not been seen. The
     * returned array is the table's internal storage; callers must
     * not mutate it directly — use {@link #update} for gradient
     * updates.
     */
    public double[] embed(String symbol) {
        return embeddings.computeIfAbsent(symbol, k -> {
            double[] v = new double[dim];
            for (int i = 0; i < dim; i++) {
                v[i] = (rng.nextDouble() * 2.0 - 1.0) * initBound;
            }
            return v;
        });
    }

    /**
     * Returns the symbol whose stored embedding is closest to
     * {@code query} by cosine similarity. Empty if the table is empty
     * or {@code query} has zero norm. Ties are broken by insertion
     * order.
     */
    public Optional<String> nearest(double[] query) {
        if (embeddings.isEmpty()) return Optional.empty();
        double qNorm = norm(query);
        if (qNorm == 0.0) return Optional.empty();

        String best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, double[]> e : embeddings.entrySet()) {
            double[] v = e.getValue();
            double vNorm = norm(v);
            if (vNorm == 0.0) continue;
            double score = dot(query, v) / (qNorm * vNorm);
            if (score > bestScore) {
                bestScore = score;
                best = e.getKey();
            }
        }
        return Optional.ofNullable(best);
    }

    /**
     * In-place gradient step on the embedding for {@code symbol}:
     * {@code emb[i] -= lr * gradient[i]}. The symbol must have been
     * previously embedded.
     */
    public void update(String symbol, double[] gradient, double lr) {
        double[] v = embeddings.get(symbol);
        if (v == null) {
            throw new IllegalStateException("no embedding for symbol: " + symbol);
        }
        if (gradient.length != dim) {
            throw new IllegalArgumentException(
                    "gradient dim " + gradient.length + " != table dim " + dim);
        }
        for (int i = 0; i < dim; i++) {
            v[i] -= lr * gradient[i];
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static double norm(double[] a) {
        double s = 0.0;
        for (double v : a) s += v * v;
        return Math.sqrt(s);
    }
}
