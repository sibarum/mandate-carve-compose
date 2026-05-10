package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: ?a + 0 → ?a.
 *
 * Inversion is non-unique (?a → ?a + 0 introduces the literal 0); the base
 * class's inferInput will fail on substitute because the LHS has a literal
 * pattern (no hole) and the RHS shape does not constrain the literal. We
 * accept the v2 limitation: this rule's inversion is identity-only — it
 * never proposes synthesizing a "+ 0" wrapper. The framework can still use
 * the rule in the forward direction.
 */
public final class IdentityZero extends RewriteRulePrimitive {
    public IdentityZero() {
        super("identity-zero",
                Pattern.add(Pattern.hole("a"), Pattern.lit(0)),
                Pattern.hole("a"));
    }
}
