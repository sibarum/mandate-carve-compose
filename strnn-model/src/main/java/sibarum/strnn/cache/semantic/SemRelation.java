package sibarum.strnn.cache.semantic;

import java.util.Objects;

/**
 * One full line from the semantic file: a dichotomy contextualized by an
 * expression. {@code (A | B) ~> rhs} says A and B are opposites along the
 * axis described by rhs.
 */
public record SemRelation(Dichotomy lhs, SemExpr rhs) {
    public SemRelation {
        Objects.requireNonNull(lhs);
        Objects.requireNonNull(rhs);
    }

    @Override
    public String toString() {
        return lhs + " ~> " + rhs;
    }
}
