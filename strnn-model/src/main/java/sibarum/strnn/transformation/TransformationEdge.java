package sibarum.strnn.transformation;

import java.util.Objects;

/**
 * Directed edge in the transformation graph. Exists when {@code from}'s output
 * type appears among {@code to}'s input types. Slot selection at the target is
 * deferred to carving time; one TransformationEdge can be used for any of the
 * type-compatible slots of the target node.
 */
public final class TransformationEdge {
    private final TransformationNode from;
    private final TransformationNode to;
    private final EdgeStats stats;

    public TransformationEdge(TransformationNode from, TransformationNode to) {
        this.from = Objects.requireNonNull(from);
        this.to = Objects.requireNonNull(to);
        if (!to.inputTypes().contains(from.outputType())) {
            throw new IllegalArgumentException(
                    "type incompatible: " + from + " -> " + to);
        }
        this.stats = new EdgeStats();
    }

    public TransformationNode from() {
        return from;
    }

    public TransformationNode to() {
        return to;
    }

    public EdgeStats stats() {
        return stats;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof TransformationEdge e && from.equals(e.from) && to.equals(e.to);
    }

    @Override
    public int hashCode() {
        return Objects.hash(from, to);
    }

    @Override
    public String toString() {
        return from.id() + " -> " + to.id();
    }
}
