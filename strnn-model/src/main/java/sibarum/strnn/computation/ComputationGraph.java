package sibarum.strnn.computation;

import sibarum.strnn.value.Value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A DAG of CompGraphNodes plus the binding for any externally-supplied root
 * values. execute() runs forward in topological order, populating each node's
 * produced value. The terminal node's produced value is the computation's
 * result.
 */
public final class ComputationGraph {
    private final List<CompGraphNode> nodes;
    private final CompGraphNode terminal;
    private final Map<RootKey, Value> rootBindings;
    private List<CompGraphNode> topoOrder;

    public ComputationGraph(List<CompGraphNode> nodes, CompGraphNode terminal) {
        this.nodes = List.copyOf(nodes);
        this.terminal = Objects.requireNonNull(terminal);
        this.rootBindings = new LinkedHashMap<>();
        if (!nodes.contains(terminal)) {
            throw new IllegalArgumentException("terminal not in node list");
        }
    }

    public void bindRoot(CompGraphNode node, int slotIndex, Value v) {
        rootBindings.put(new RootKey(node, slotIndex), v);
    }

    public List<CompGraphNode> nodes() {
        return nodes;
    }

    public CompGraphNode terminal() {
        return terminal;
    }

    public List<CompGraphNode> topoOrder() {
        if (topoOrder == null) topoOrder = Collections.unmodifiableList(computeTopo());
        return topoOrder;
    }

    public Value execute() {
        for (CompGraphNode n : topoOrder()) {
            List<Value> inputs = new ArrayList<>(n.slotCount());
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource src = n.slot(i);
                if (src == null) {
                    Value v = rootBindings.get(new RootKey(n, i));
                    if (v == null) {
                        throw new IllegalStateException(
                                "node " + n.id() + " slot " + i + " has no source and no root binding");
                    }
                    inputs.add(v);
                } else {
                    Value v = src.source().producedValue();
                    if (v == null) {
                        throw new IllegalStateException(
                                "source " + src.source().id() + " not yet produced when computing " + n.id());
                    }
                    inputs.add(v);
                }
            }
            n.setProducedValue(n.tNode().primitive().apply(inputs));
        }
        return terminal.producedValue();
    }

    private List<CompGraphNode> computeTopo() {
        Map<CompGraphNode, Integer> indeg = new HashMap<>();
        Map<CompGraphNode, List<CompGraphNode>> children = new HashMap<>();
        for (CompGraphNode n : nodes) {
            indeg.put(n, 0);
            children.put(n, new ArrayList<>());
        }
        for (CompGraphNode n : nodes) {
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource s = n.slot(i);
                if (s != null) {
                    indeg.merge(n, 1, Integer::sum);
                    children.get(s.source()).add(n);
                }
            }
        }
        List<CompGraphNode> sorted = new ArrayList<>();
        List<CompGraphNode> ready = new ArrayList<>();
        for (Map.Entry<CompGraphNode, Integer> e : indeg.entrySet()) {
            if (e.getValue() == 0) ready.add(e.getKey());
        }
        Set<CompGraphNode> seen = new HashSet<>();
        while (!ready.isEmpty()) {
            CompGraphNode n = ready.removeLast();
            if (!seen.add(n)) continue;
            sorted.add(n);
            for (CompGraphNode c : children.get(n)) {
                int d = indeg.merge(c, -1, Integer::sum);
                if (d == 0) ready.add(c);
            }
        }
        if (sorted.size() != nodes.size()) {
            throw new IllegalStateException("computation graph has a cycle");
        }
        return sorted;
    }

    /**
     * Returns true iff {@code ancestor} can reach {@code descendant} via
     * directed slot wires. Used by the mandate verifier for ordering checks.
     */
    public boolean reaches(CompGraphNode ancestor, CompGraphNode descendant) {
        if (ancestor.equals(descendant)) return true;
        Set<CompGraphNode> visited = new HashSet<>();
        List<CompGraphNode> stack = new ArrayList<>();
        stack.add(ancestor);
        Map<CompGraphNode, List<CompGraphNode>> children = new HashMap<>();
        for (CompGraphNode n : nodes) children.put(n, new ArrayList<>());
        for (CompGraphNode n : nodes) {
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource s = n.slot(i);
                if (s != null) children.get(s.source()).add(n);
            }
        }
        while (!stack.isEmpty()) {
            CompGraphNode cur = stack.removeLast();
            if (!visited.add(cur)) continue;
            if (cur.equals(descendant)) return true;
            stack.addAll(children.get(cur));
        }
        return false;
    }

    private record RootKey(CompGraphNode node, int slotIndex) {
    }
}
