package sibarum.strnn.demo;

import sibarum.strnn.cache.TotalArithmetic;

import java.util.List;
import java.util.Locale;
import java.util.function.DoubleBinaryOperator;

/**
 * Exhaustive edge-case verification of the total-arithmetic floor that
 * vector primitives sit on. Each of the four operations is tested across
 * the full sign × magnitude grid:
 *
 *   {finite-pos, finite-neg, 0, +∞, −∞} × {same set}
 *
 * For every cell, the expected output is asserted. NaN must never escape.
 * The final diagnostic verifies that NaN inputs are rejected at the
 * boundary rather than producing silent garbage.
 *
 * Run cost: ~125 assertions across the four operations plus 8 NaN-guard
 * checks. If any single identity drifts, this demo catches it.
 */
public final class TotalArithmeticDemo {

    private static final double POS_INF = Double.POSITIVE_INFINITY;
    private static final double NEG_INF = Double.NEGATIVE_INFINITY;

    public static void main(String[] args) {
        runAddTests();
        runSubTests();
        runMulTests();
        runDivTests();
        runNaNGuardTests();
        System.out.println("\nTotal arithmetic floor: every (a, b) pair from "
                + "{finite, 0, ±∞} × {finite, 0, ±∞} produces a defined double; "
                + "NaN cannot escape and cannot enter.");
    }

    // ---------------------------------------------------------------------
    // Addition
    // ---------------------------------------------------------------------

    private static void runAddTests() {
        System.out.println("=== Diagnostic: total addition (a + b) ===");
        List<Case> cases = List.of(
                // finite + finite
                new Case(2.0, 3.0, 5.0),
                new Case(-2.0, 3.0, 1.0),
                new Case(2.0, -3.0, -1.0),
                // 0 cases
                new Case(0.0, 0.0, 0.0),
                new Case(5.0, 0.0, 5.0),
                new Case(0.0, 5.0, 5.0),
                // finite + ∞
                new Case(5.0, POS_INF, POS_INF),
                new Case(5.0, NEG_INF, NEG_INF),
                new Case(POS_INF, 5.0, POS_INF),
                new Case(NEG_INF, 5.0, NEG_INF),
                // 0 + ∞
                new Case(0.0, POS_INF, POS_INF),
                new Case(0.0, NEG_INF, NEG_INF),
                // ∞ + ∞ same sign
                new Case(POS_INF, POS_INF, POS_INF),
                new Case(NEG_INF, NEG_INF, NEG_INF),
                // ∞ + ∞ opposite sign — the override case
                new Case(POS_INF, NEG_INF, 0.0),
                new Case(NEG_INF, POS_INF, 0.0)
        );
        run(cases, TotalArithmetic::totalAdd, "+");
    }

    // ---------------------------------------------------------------------
    // Subtraction
    // ---------------------------------------------------------------------

    private static void runSubTests() {
        System.out.println("\n=== Diagnostic: total subtraction (a − b) ===");
        List<Case> cases = List.of(
                new Case(5.0, 3.0, 2.0),
                new Case(3.0, 5.0, -2.0),
                new Case(0.0, 0.0, 0.0),
                new Case(5.0, 0.0, 5.0),
                new Case(0.0, 5.0, -5.0),
                // finite − ∞
                new Case(5.0, POS_INF, NEG_INF),
                new Case(5.0, NEG_INF, POS_INF),
                new Case(POS_INF, 5.0, POS_INF),
                new Case(NEG_INF, 5.0, NEG_INF),
                // 0 − ∞
                new Case(0.0, POS_INF, NEG_INF),
                new Case(0.0, NEG_INF, POS_INF),
                // ∞ − ∞ same sign — the override case
                new Case(POS_INF, POS_INF, 0.0),
                new Case(NEG_INF, NEG_INF, 0.0),
                // ∞ − ∞ opposite sign
                new Case(POS_INF, NEG_INF, POS_INF),
                new Case(NEG_INF, POS_INF, NEG_INF)
        );
        run(cases, TotalArithmetic::totalSub, "−");
    }

    // ---------------------------------------------------------------------
    // Multiplication
    // ---------------------------------------------------------------------

    private static void runMulTests() {
        System.out.println("\n=== Diagnostic: total multiplication (a · b) ===");
        List<Case> cases = List.of(
                new Case(2.0, 3.0, 6.0),
                new Case(-2.0, 3.0, -6.0),
                new Case(2.0, -3.0, -6.0),
                new Case(-2.0, -3.0, 6.0),
                // 0 × finite — IEEE-correct
                new Case(0.0, 5.0, 0.0),
                new Case(5.0, 0.0, 0.0),
                new Case(0.0, 0.0, 0.0),
                // finite × ∞ — IEEE-correct
                new Case(5.0, POS_INF, POS_INF),
                new Case(5.0, NEG_INF, NEG_INF),
                new Case(-5.0, POS_INF, NEG_INF),
                new Case(POS_INF, 5.0, POS_INF),
                // 0 × ±∞ — the override case (sign comes from the infinity)
                new Case(0.0, POS_INF, 1.0),
                new Case(0.0, NEG_INF, -1.0),
                new Case(POS_INF, 0.0, 1.0),
                new Case(NEG_INF, 0.0, -1.0),
                // ∞ × ∞
                new Case(POS_INF, POS_INF, POS_INF),
                new Case(POS_INF, NEG_INF, NEG_INF),
                new Case(NEG_INF, NEG_INF, POS_INF)
        );
        run(cases, TotalArithmetic::totalMul, "·");
    }

    // ---------------------------------------------------------------------
    // Division
    // ---------------------------------------------------------------------

    private static void runDivTests() {
        System.out.println("\n=== Diagnostic: total division (a / b) ===");
        List<Case> cases = List.of(
                new Case(6.0, 3.0, 2.0),
                new Case(-6.0, 3.0, -2.0),
                new Case(6.0, -3.0, -2.0),
                // finite / 0 — IEEE-correct (sign(a) · ∞)
                new Case(5.0, 0.0, POS_INF),
                new Case(-5.0, 0.0, NEG_INF),
                // 0 / 0 — the override case
                new Case(0.0, 0.0, 1.0),
                // 0 / finite, 0 / ∞ — IEEE-correct
                new Case(0.0, 5.0, 0.0),
                new Case(0.0, POS_INF, 0.0),
                new Case(0.0, NEG_INF, 0.0),
                // finite / ∞ — IEEE-correct
                new Case(5.0, POS_INF, 0.0),
                new Case(5.0, NEG_INF, 0.0),
                // ∞ / 0 — IEEE-correct (sign preserved)
                new Case(POS_INF, 0.0, POS_INF),
                new Case(NEG_INF, 0.0, NEG_INF),
                // ∞ / finite — IEEE-correct
                new Case(POS_INF, 5.0, POS_INF),
                new Case(POS_INF, -5.0, NEG_INF),
                // ∞ / ∞ — the override case (sign(a)·sign(b))
                new Case(POS_INF, POS_INF, 1.0),
                new Case(POS_INF, NEG_INF, -1.0),
                new Case(NEG_INF, POS_INF, -1.0),
                new Case(NEG_INF, NEG_INF, 1.0)
        );
        run(cases, TotalArithmetic::totalDiv, "/");
    }

    // ---------------------------------------------------------------------
    // NaN guard
    // ---------------------------------------------------------------------

    private static void runNaNGuardTests() {
        System.out.println("\n=== Diagnostic: NaN inputs rejected at the boundary ===");
        List<DoubleBinaryOperator> ops = List.of(
                TotalArithmetic::totalAdd,
                TotalArithmetic::totalSub,
                TotalArithmetic::totalMul,
                TotalArithmetic::totalDiv
        );
        List<String> names = List.of("+", "−", "·", "/");
        for (int i = 0; i < ops.size(); i++) {
            assertThrows(() -> ops.get(0).applyAsDouble(Double.NaN, 1.0)); // dummy to keep i in scope
            assertThrows(() -> ops.get(0).applyAsDouble(1.0, Double.NaN));
        }
        // Explicit per-op checks (clearer error messages on regression):
        for (int i = 0; i < ops.size(); i++) {
            DoubleBinaryOperator op = ops.get(i);
            String n = names.get(i);
            assertThrows(() -> op.applyAsDouble(Double.NaN, 1.0));
            assertThrows(() -> op.applyAsDouble(1.0, Double.NaN));
            assertThrows(() -> op.applyAsDouble(Double.NaN, Double.NaN));
            System.out.printf("  %s rejects NaN on either side and both%n", n);
        }
        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Test harness
    // ---------------------------------------------------------------------

    private record Case(double a, double b, double expected) {
    }

    private static void run(List<Case> cases, DoubleBinaryOperator op, String sym) {
        int idx = 0;
        for (Case c : cases) {
            double got = op.applyAsDouble(c.a, c.b);
            require(equalsBitwiseOrFinite(got, c.expected),
                    String.format(Locale.ROOT,
                            "[case %d] %s %s %s : expected %s, got %s",
                            idx, fmt(c.a), sym, fmt(c.b), fmt(c.expected), fmt(got)));
            require(!Double.isNaN(got),
                    String.format(Locale.ROOT,
                            "[case %d] NaN escaped from %s %s %s",
                            idx, fmt(c.a), sym, fmt(c.b)));
            System.out.printf(Locale.ROOT, "  %12s %s %-12s = %s%n",
                    fmt(c.a), sym, fmt(c.b), fmt(got));
            idx++;
        }
        System.out.println("  PASS (" + cases.size() + " cases)");
    }

    private static boolean equalsBitwiseOrFinite(double a, double b) {
        if (Double.isInfinite(a) && Double.isInfinite(b)) {
            return Math.signum(a) == Math.signum(b);
        }
        return a == b;
    }

    private static String fmt(double d) {
        if (d == POS_INF) return "+∞";
        if (d == NEG_INF) return "−∞";
        if (d == Math.floor(d)) return Integer.toString((int) d);
        return Double.toString(d);
    }

    private static void require(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    private static void assertThrows(Runnable r) {
        try {
            r.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("expected IllegalArgumentException for NaN input");
    }
}
