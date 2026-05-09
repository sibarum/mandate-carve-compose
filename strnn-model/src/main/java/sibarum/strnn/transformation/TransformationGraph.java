package sibarum.strnn.transformation;

import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Slowly-shaped substrate of the framework. Holds the set of TransformationNodes
 * and TransformationEdges and supports the lookups the carver needs.
 */
public final class TransformationGraph {
    private final Map<String, TransformationNode> nodes;
    private final List<TransformationEdge> edges;
    private final Map<TransformationNode, List<TransformationEdge>> outgoing;
    private final Map<TransformationNode, List<TransformationEdge>> incoming;
    private final Map<ValueType, List<TransformationNode>> nodesByOutput;

    public TransformationGraph(
            Collection<TransformationNode> nodeCollection,
            Collection<TransformationEdge> edgeCollection) {
        this.nodes = new HashMap<>();
        for (TransformationNode n : nodeCollection) nodes.put(n.id(), n);
        this.edges = new ArrayList<>(edgeCollection);
        this.outgoing = new HashMap<>();
        this.incoming = new HashMap<>();
        this.nodesByOutput = new HashMap<>();
        for (TransformationNode n : nodes.values()) {
            outgoing.put(n, new ArrayList<>());
            incoming.put(n, new ArrayList<>());
            nodesByOutput.computeIfAbsent(n.outputType(), k -> new ArrayList<>()).add(n);
        }
        for (TransformationEdge e : edges) {
            outgoing.get(e.from()).add(e);
            incoming.get(e.to()).add(e);
        }
    }

    public TransformationNode node(String id) {
        TransformationNode n = nodes.get(id);
        if (n == null) throw new IllegalArgumentException("no node: " + id);
        return n;
    }

    public Collection<TransformationNode> nodes() {
        return Collections.unmodifiableCollection(nodes.values());
    }

    public List<TransformationEdge> edges() {
        return Collections.unmodifiableList(edges);
    }

    public List<TransformationEdge> outgoing(TransformationNode from) {
        return outgoing.getOrDefault(from, List.of());
    }

    public List<TransformationEdge> incoming(TransformationNode to) {
        return incoming.getOrDefault(to, List.of());
    }

    public List<TransformationNode> nodesProducing(ValueType type) {
        return nodesByOutput.getOrDefault(type, List.of());
    }

    /**
     * Find the edge from -> to, if it exists. Returns null otherwise.
     */
    public TransformationEdge edge(TransformationNode from, TransformationNode to) {
        for (TransformationEdge e : outgoing.getOrDefault(from, List.of())) {
            if (e.to().equals(to)) return e;
        }
        return null;
    }
}
