package sibarum.mcc.graph.substrate;

import sibarum.mcc.primitive.Primitive;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds a {@link TransformationGraph} as "any-to-any modulo type
 * compatibility" from a primitive library: every pair {@code (A, B)}
 * where {@code A.outputType()} is in {@code B.inputTypes()} gets a
 * {@link TransformationEdge}. Type incompatibility is the only
 * omission applied at this stage; further omissions can be subtracted
 * later by the GUI or by a curator.
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
