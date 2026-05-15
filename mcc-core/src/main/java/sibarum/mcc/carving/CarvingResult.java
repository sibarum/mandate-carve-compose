package sibarum.mcc.carving;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.substrate.TransformationEdge;
import sibarum.mcc.value.Value;

import java.util.List;
import java.util.Map;

/**
 * Result of a single carving attempt. Holds the produced
 * {@link ComputationGraph}, the list of substrate edges traversed
 * (for credit assignment), the carver's per-node simulated values
 * (the value the carver expected each node to produce — used as
 * training targets for trainables placed at that node), and the root
 * bindings the carver registered.
 */
public record CarvingResult(
        ComputationGraph graph,
        List<TransformationEdge> tracedEdges,
        Map<CompGraphNode, Value> simulatedValues,
        List<RootBinding> rootBindings) {
}
