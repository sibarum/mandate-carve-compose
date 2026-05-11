package sibarum.elden.embedding;

import sibarum.strnn.cache.SymbolEmbeddingTable;

import java.util.ArrayList;
import java.util.List;

/**
 * Deterministic sliding-window context encoder. For each position {@code i}
 * in a token sequence, the context vector is the concatenation of token
 * embeddings within a window of {@code +/- windowRadius} positions. Missing
 * neighbors (off the start or end of the sequence) contribute zero vectors.
 *
 * Output dimensionality per position is {@code (2 * windowRadius + 1) * dim}.
 * The center embedding sits in the middle slot — this matters for any
 * downstream classifier that wants to read it positionally.
 *
 * No learnable parameters. Pure preprocessing.
 */
public final class ContextEncoder {

    private final SymbolEmbeddingTable table;
    private final int windowRadius;

    public ContextEncoder(SymbolEmbeddingTable table, int windowRadius) {
        if (windowRadius < 0) throw new IllegalArgumentException("windowRadius must be >= 0");
        this.table = table;
        this.windowRadius = windowRadius;
    }

    public int contextDim() {
        return (2 * windowRadius + 1) * table.dim();
    }

    /** Context vector for one position in a token sequence. */
    public double[] encode(List<String> tokens, int position) {
        int dim = table.dim();
        double[] out = new double[contextDim()];
        for (int offset = -windowRadius; offset <= windowRadius; offset++) {
            int idx = position + offset;
            int slot = offset + windowRadius;
            if (idx >= 0 && idx < tokens.size()) {
                double[] emb = table.embed(tokens.get(idx));
                System.arraycopy(emb, 0, out, slot * dim, dim);
            }
            // else: zero vector — already what's there from `new double[]`.
        }
        return out;
    }

    /** Context vectors for every position in a sequence. */
    public List<double[]> encodeAll(List<String> tokens) {
        List<double[]> out = new ArrayList<>(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            out.add(encode(tokens, i));
        }
        return out;
    }
}
