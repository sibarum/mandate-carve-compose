package sibarum.strnn.cache;

/**
 * What a KV cache stores per key. Today the only impl is
 * {@link EmbeddingItem} (a stored vector). The interface exists to hold the
 * type slot open for future variants — most notably {@code NetworkItem},
 * which would let a cache store carved+trained subgraphs as values. That's
 * the Key-Network milestone.
 *
 * Existing code that only knows about embeddings can continue to talk to
 * {@link SymbolEmbeddingTable} directly. New code that wants to be
 * forward-compatible across item types can use {@code CachedItem} as the
 * common supertype.
 */
public sealed interface CachedItem permits EmbeddingItem {
}
