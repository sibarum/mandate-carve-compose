package sibarum.strnn.transformation;

import sibarum.strnn.primitive.Primitive;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds a TransformationGraph as &quot;any-to-any modulo type compatibility&quot;
 * from a primitive library: every pair (A, B) where A.outputType() is in
 * B.inputTypes() gets a TransformationEdge. Type incompatibility is the only
 * omission applied at this stage; further omissions could be subtracted later.
 */
public final class TransformationGraphBuilder {
    private final Map<String, TransformationNode> nodes = new HashMap<>();

    public TransformationGraphBuilder addNode(String id, Primitive primitive) {
        TransformationNode node = new TransformationNode(id, primitive);
        if (nodes.put(id, node) != null) {
            throw new IllegalArgumentException("duplicate node id: " + id);
        }
        return this;
    }

    public Collection<TransformationNode> nodes() {
        return nodes.values();
    }

    public TransformationGraph build() {
        List<TransformationEdge> edges = new ArrayList<>();
        for (TransformationNode from : nodes.values()) {
            for (TransformationNode to : nodes.values()) {
                if (from.equals(to)) continue;
                if (to.inputTypes().contains(from.outputType())) {
                    edges.add(new TransformationEdge(from, to));
                }
            }
        }
        return new TransformationGraph(nodes.values(), edges);
    }
}
