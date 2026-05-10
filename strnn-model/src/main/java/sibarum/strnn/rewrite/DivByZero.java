package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: {@code ?a / 0 → ?a · ω}.
 *
 * In a total, reversible algebra, division by zero produces ω (the
 * multiplicative inverse of 0); coefficients carry through unchanged, just as
 * they would for division by any other variable. Concretely:
 *
 *   1 / 0  →  1 · ω
 *   2 / 0  →  2 · ω
 *   x / 0  →  x · ω
 *
 * Whether {@code 1·ω} should be further reduced to {@code ω} is a separate
 * rule (a left-sided multiplicative-identity simplification); each rule does
 * one thing, and composition is the carver's job.
 *
 * Reversible: {@code inferInput} on {@code ?a · ω} recovers {@code ?a / 0}.
 */
public final class DivByZero extends RewriteRulePrimitive implements Reversible {
    public DivByZero() {
        super("div-by-zero",
                Pattern.div(Pattern.hole("a"), Pattern.lit(0)),
                Pattern.mul(Pattern.hole("a"), Pattern.omega()));
    }
}
