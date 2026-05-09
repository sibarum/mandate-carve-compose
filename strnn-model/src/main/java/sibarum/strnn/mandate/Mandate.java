package sibarum.strnn.mandate;

import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueDistance;

import java.util.Objects;

/**
 * §6 mandate: a typed expected value the carved computation must produce
 * <i>somewhere</i> in its DAG. Non-local: the verifier searches every node's
 * produced value for a match.
 *
 * @param name        identifier used in reports (e.g. "tokens", "intermediate_product")
 * @param expected    the value that must be produced
 * @param tolerance   slack passed to ValueDistance.matches; structural-equality types ignore this
 * @param ordering    monotonic tag; mandates with smaller ordering must be produced
 *                    by a topological ancestor of mandates with larger ordering
 * @param isResult    true iff this mandate must be matched specifically by the
 *                    terminal node (the computation's output)
 */
public record Mandate(String name, Value expected, double tolerance, int ordering, boolean isResult) {

    public Mandate {
        Objects.requireNonNull(name);
        Objects.requireNonNull(expected);
    }

    public boolean matches(Value actual) {
        return ValueDistance.matches(expected, actual, tolerance);
    }

    public static Mandate intermediate(String name, Value expected, double tol, int ordering) {
        return new Mandate(name, expected, tol, ordering, false);
    }

    public static Mandate result(Value expected, double tol, int ordering) {
        return new Mandate("result", expected, tol, ordering, true);
    }
}
