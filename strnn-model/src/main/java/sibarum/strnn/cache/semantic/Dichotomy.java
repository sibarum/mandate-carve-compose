package sibarum.strnn.cache.semantic;

import java.util.Objects;

/**
 * A single dichotomy: two atoms that anchor opposite ends of a semantic axis.
 * Parsed from the {@code (A | B)} form on the LHS of every relation.
 */
public record Dichotomy(String left, String right) {
    public Dichotomy {
        Objects.requireNonNull(left);
        Objects.requireNonNull(right);
    }

    @Override
    public String toString() {
        return "(" + left + " | " + right + ")";
    }
}
