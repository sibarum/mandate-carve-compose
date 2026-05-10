package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: 0 · ω → 1.
 *
 * Structurally the same identity as {@code 0/0 → 1}: division is multiplication
 * by inverse, and ω is the multiplicative inverse of 0. Encoded directly so
 * the carver does not need to compose div-by-zero with subsequent rewrites
 * to reach the simplest form.
 *
 * Sided narrowly: matches {@code 0 · ω} only. The mirrored form {@code ω · 0}
 * would need a separate rule (commutativity is not built into the matcher,
 * by design — different operand orderings may have different ranks under the
 * carver's edge statistics).
 */
public final class ZeroTimesOmega extends RewriteRulePrimitive implements Reversible {
    public ZeroTimesOmega() {
        super("zero-times-omega",
                Pattern.mul(Pattern.lit(0), Pattern.omega()),
                Pattern.lit(1));
    }
}
