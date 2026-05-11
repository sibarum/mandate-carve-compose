package sibarum.strnn.carving;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.value.Value;

/**
 * Records that the carver bound a particular {@link CompGraphNode}'s input
 * slot to a value supplied externally (typically the root input). Stored in
 * {@link CarvingResult} so callers can rebind those slots when reusing a
 * carved graph with new inputs — most notably {@code NetworkItem}, which
 * cycles a trained carving against fresh queries on every invocation.
 */
public record RootBinding(CompGraphNode node, int slot, Value value) {
}
