package sibarum.mcc.graph;

import sibarum.mcc.graph.substrate.TransformationEdge;

/**
 * A wire in the computation graph: the source {@link CompGraphNode}
 * whose output flows into a particular input slot, plus the
 * originating {@link TransformationEdge} so a trainer can credit the
 * correct substrate edge for the value's contribution.
 */
public record SlotSource(CompGraphNode source, TransformationEdge originatingEdge) {
}
