package sibarum.strnn.store;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Read-only view of a keyed collection of vectors with approximate
 * nearest-neighbor lookup.
 *
 * Implementations vary by index strategy:
 *   - An in-memory flat-array implementation does exhaustive cosine /
 *     Euclidean comparison — fine for vocabularies up to ~tens of thousands.
 *   - A {@code JVector}-backed implementation maintains an HNSW index on
 *     disk and serves nearest-neighbor queries in sub-linear time.
 *
 * All vectors in a single store have the same dimension. Distances are
 * implementation-defined (typically cosine or squared Euclidean) but
 * monotonically consistent — smaller means more similar.
 */
public interface VectorStore {

    Optional<float[]> get(String id);

    boolean contains(String id);

    Collection<String> ids();

    int size();

    int dimension();

    /** Top {@code k} nearest neighbors to a free-form query vector. */
    List<NearestNeighbor> nearest(float[] query, int k);

    /** Top {@code k} nearest neighbors to a stored vector, excluding itself. */
    List<NearestNeighbor> nearestById(String id, int k);

    record NearestNeighbor(String id, float distance) {}
}
