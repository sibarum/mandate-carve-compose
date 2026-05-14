package sibarum.mcc.mandate;

import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueDistance;

import java.util.Objects;

/**
 * Declarative spec of a typed expected value that the executed graph
 * must produce <em>somewhere</em> in its DAG. Non-local: the verifier
 * searches every node's produced value for a match.
 *
 * @param name        identifier used in reports (e.g. "intermediate", "result")
 * @param expected    the value that must be produced
 * @param tolerance   slack passed to {@link ValueDistance#matches}; structural-equality types ignore this
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
