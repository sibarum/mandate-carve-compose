package sibarum.strnn.computation;

import sibarum.strnn.transformation.TransformationEdge;

/**
 * A wire in the computation graph: the source CompGraphNode whose output flows
 * into a particular input slot, plus the originating TransformationEdge so the
 * trainer can credit / debit the correct prior-graph edge.
 */
public record SlotSource(CompGraphNode source, TransformationEdge originatingEdge) {
}
