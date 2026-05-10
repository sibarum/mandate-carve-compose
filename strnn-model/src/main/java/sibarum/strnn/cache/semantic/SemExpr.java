package sibarum.strnn.cache.semantic;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * The right-hand side of a semantic relation: a tree of atoms combined by
 * the file's six operators (excluding the dichotomy {@code |} and the top-level
 * {@code ~>}, which live above this type).
 *
 * Operator precedence, tightest to loosest:
 *   1. {@code :}  qualifier — {@code physical:state} reads as one unit
 *   2. {@code +} / {@code *}  conjunction / composition (same level, left-assoc)
 *   3. {@code &}  union (left-assoc, flattened into a single Union node)
 *
 * Atoms are bare identifiers (the words in the ontology). Parens override
 * precedence and produce no node of their own — they're transparent in the AST.
 */
public sealed interface SemExpr {

    record Atom(String name) implements SemExpr {
        public Atom {
            Objects.requireNonNull(name);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /** A:B — B is a facet/aspect of A. */
    record Qualified(SemExpr head, SemExpr facet) implements SemExpr {
        public Qualified {
            Objects.requireNonNull(head);
            Objects.requireNonNull(facet);
        }

        @Override
        public String toString() {
            return head + ":" + facet;
        }
    }

    /** A * B — composed/inseparable. */
    record Composition(SemExpr left, SemExpr right) implements SemExpr {
        public Composition {
            Objects.requireNonNull(left);
            Objects.requireNonNull(right);
        }

        @Override
        public String toString() {
            return "(" + left + " * " + right + ")";
        }
    }

    /** A + B — both required, distinct. */
    record Conjunction(SemExpr left, SemExpr right) implements SemExpr {
        public Conjunction {
            Objects.requireNonNull(left);
            Objects.requireNonNull(right);
        }

        @Override
        public String toString() {
            return "(" + left + " + " + right + ")";
        }
    }

    /** A & B & C — any/all members of the union apply. Flattened. */
    record Union(List<SemExpr> members) implements SemExpr {
        public Union {
            Objects.requireNonNull(members);
            if (members.size() < 2) {
                throw new IllegalArgumentException(
                        "Union must have at least two members; got " + members.size());
            }
            members = List.copyOf(members);
        }

        @Override
        public String toString() {
            return "(" + members.stream().map(Object::toString).collect(Collectors.joining(" & ")) + ")";
        }
    }
}
