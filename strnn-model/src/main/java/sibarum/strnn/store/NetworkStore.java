package sibarum.strnn.store;

import sibarum.strnn.cache.NetworkItem;

import java.util.Collection;
import java.util.Optional;

/**
 * Read-only view of a keyed collection of {@link NetworkItem}s.
 *
 * Implementations vary by where the network weights live:
 *   - {@link sibarum.strnn.cache.NetworkCache} keeps everything in memory and
 *     spawns items via training on demand.
 *   - A future disk-backed implementation might memory-map LMDB or stream
 *     weights from a SQLite BLOB column.
 *
 * Callers that only need to read networks (the carver, inference paths)
 * should depend on this interface rather than on {@code NetworkCache}
 * directly, so the storage backend can be swapped without rippling through
 * the rest of the framework.
 */
public interface NetworkStore {

    Optional<NetworkItem> get(String key);

    boolean contains(String key);

    Collection<String> keys();

    int size();
}
