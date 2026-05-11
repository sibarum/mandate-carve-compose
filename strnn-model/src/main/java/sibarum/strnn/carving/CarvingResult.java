package sibarum.strnn.carving;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.value.Value;

import java.util.List;
import java.util.Map;

/**
 * Result of a single carving attempt. Holds the produced ComputationGraph,
 * the list of TransformationEdges traversed during construction (for credit
 * assignment), the carver's simulated values per CompGraphNode (the value
 * the carver expected each node to produce — used as the supervision target
 * for any trainable primitives placed at that node), and the root bindings
 * the carver registered (so callers can rebind those slots when reusing the
 * carving with different inputs).
 */
public record CarvingResult(
        ComputationGraph graph,
        List<TransformationEdge> tracedEdges,
        Map<CompGraphNode, Value> simulatedValues,
        List<RootBinding> rootBindings) {
}
