package sibarum.mcc.graph.substrate;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.ValueType;

import java.util.List;
import java.util.Objects;

/**
 * A node in the transformation graph (the GUI palette substrate): an
 * addressable identity wrapping a {@link Primitive}. Multiple
 * {@code TransformationNode}s may share the same primitive instance
 * — e.g. two MLP nodes backed by the same network share weights.
 */
public final class TransformationNode {
    private final String id;
    private final Primitive primitive;

    public TransformationNode(String id, Primitive primitive) {
        this.id = Objects.requireNonNull(id);
        this.primitive = Objects.requireNonNull(primitive);
    }

    public String id() { return id; }
    public Primitive primitive() { return primitive; }

    public List<ValueType> inputTypes() {
        return primitive.inputTypes();
    }

    public ValueType outputType() {
        return primitive.outputType();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof TransformationNode n && id.equals(n.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return id + "[" + primitive.name() + "]";
    }
}
