package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: 0 / 0 → 1.
 *
 * Standard algebra leaves 0/0 indeterminate; the total, reversible algebra
 * picks 1, consistent with reading 0/0 as 0·(1/0) = 0·ω, and 0·ω → 1 as the
 * dual of 1/0 → ω. This rule encodes that choice directly so the carver does
 * not need to compose two rules to reach it on the simplest case.
 */
public final class ZeroDivZero extends RewriteRulePrimitive implements Reversible {
    public ZeroDivZero() {
        super("zero-div-zero",
                Pattern.div(Pattern.lit(0), Pattern.lit(0)),
                Pattern.lit(1));
    }
}
