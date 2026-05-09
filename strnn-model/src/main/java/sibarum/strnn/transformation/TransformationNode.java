package sibarum.strnn.transformation;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Objects;

/**
 * A node in the transformation graph: an addressable identity wrapping a
 * Primitive. Multiple TransformationNodes may share the same Primitive
 * instance (e.g. two MlpPrimitives backed by the same Mlp share weights).
 */
public final class TransformationNode {
    private final String id;
    private final Primitive primitive;

    public TransformationNode(String id, Primitive primitive) {
        this.id = Objects.requireNonNull(id);
        this.primitive = Objects.requireNonNull(primitive);
    }

    public String id() {
        return id;
    }

    public Primitive primitive() {
        return primitive;
    }

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
