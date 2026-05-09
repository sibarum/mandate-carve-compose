package sibarum.strnn.computation;

import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Instance of a TransformationNode appearing in a particular ComputationGraph.
 * The same TransformationNode may appear as multiple CompGraphNodes within one
 * carving (the SAME instance has the SAME identity). {@code slots} maps each
 * input slot of the underlying primitive to its source — a SlotSource pointing
 * at another CompGraphNode, or null if the value comes from an external root
 * binding.
 */
public final class CompGraphNode {
    private final String id;
    private final TransformationNode tNode;
    private final SlotSource[] slots;
    private Value producedValue;

    public CompGraphNode(String id, TransformationNode tNode) {
        this.id = Objects.requireNonNull(id);
        this.tNode = Objects.requireNonNull(tNode);
        this.slots = new SlotSource[tNode.inputTypes().size()];
    }

    public String id() {
        return id;
    }

    public TransformationNode tNode() {
        return tNode;
    }

    public int slotCount() {
        return slots.length;
    }

    public SlotSource slot(int i) {
        return slots[i];
    }

    public List<SlotSource> slotsCopy() {
        return new ArrayList<>(Arrays.asList(slots));
    }

    public void wire(int slotIndex, SlotSource source) {
        if (slots[slotIndex] != null) {
            throw new IllegalStateException("slot " + slotIndex + " already wired on " + id);
        }
        slots[slotIndex] = source;
    }

    public void clearSlot(int slotIndex) {
        slots[slotIndex] = null;
    }

    public Value producedValue() {
        return producedValue;
    }

    public void setProducedValue(Value v) {
        this.producedValue = v;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof CompGraphNode n && id.equals(n.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return id + "(" + tNode.primitive().name() + ")";
    }
}
