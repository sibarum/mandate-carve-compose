package sibarum.mcc.carving;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.value.Value;

/**
 * Records that the carver bound a particular {@link CompGraphNode}'s
 * input slot to a value supplied externally (the root input). Stored
 * in {@link CarvingResult} so callers can rebind the slot with fresh
 * input when reusing the carving.
 */
public record RootBinding(CompGraphNode node, int slot, Value value) {
}
