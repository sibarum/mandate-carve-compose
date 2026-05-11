package sibarum.strnn.store;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Read-only view of a typed entity-and-relation graph.
 *
 * Generic over node type {@code N} and relation type {@code R} so that domain
 * code (e.g. the Elden Ring corpus) can supply its own concrete records
 * without dragging them into the framework. The framework only assumes that
 * nodes are addressable by string id and relations carry source/target ids.
 *
 * Implementations vary by where the graph lives:
 *   - An in-memory implementation can wrap an existing
 *     {@code Map<String, N>} plus {@code List<R>}.
 *   - A disk-backed implementation might use SQLite (nodes table + edges
 *     table + recursive CTE for traversal) or an embedded property-graph DB.
 *
 * Mutation is out of scope for this interface — domain code typically writes
 * directly to the concrete in-memory record and only depends on this
 * interface from the read side.
 */
public interface EntityStore<N, R> {

    Optional<N> getNode(String id);

    boolean containsNode(String id);

    Collection<String> nodeIds();

    int nodeCount();

    List<R> relations();

    List<R> outgoing(String fromId);

    List<R> incoming(String toId);

    int relationCount();
}
