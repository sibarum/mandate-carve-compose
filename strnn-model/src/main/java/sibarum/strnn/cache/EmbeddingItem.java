package sibarum.strnn.cache;

import java.util.Objects;

/**
 * A stored vector in a KV cache. Wraps a {@code double[]} and exposes it as
 * a {@link CachedItem}. Currently the only implementation; sits next to a
 * future {@code NetworkItem} that would store a trained subgraph instead of
 * a vector (the Key-Network landing).
 *
 * The vector array is held by reference — callers must not mutate it after
 * construction unless they own the cache and intend the mutation as a
 * gradient update.
 */
public record EmbeddingItem(double[] vector) implements CachedItem {
    public EmbeddingItem {
        Objects.requireNonNull(vector);
    }

    public int dim() {
        return vector.length;
    }
}
