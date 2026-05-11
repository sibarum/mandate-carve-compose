package sibarum.strnn.cache;

/**
 * What a KV cache stores per key. Two implementations:
 *   - {@link EmbeddingItem} — a stored vector (the original KV).
 *   - {@link NetworkItem}   — a stored trained subgraph, retrieved as a
 *                              callable function rather than a returned
 *                              value (the Key-Network landing).
 *
 * Existing code that only knows about embeddings can continue to talk to
 * {@link SymbolEmbeddingTable} directly. Code that wants to be
 * forward-compatible across item types uses {@code CachedItem} as the
 * common supertype.
 */
public sealed interface CachedItem permits EmbeddingItem, NetworkItem {
}
