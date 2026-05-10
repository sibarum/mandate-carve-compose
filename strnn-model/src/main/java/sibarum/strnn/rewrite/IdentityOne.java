package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: ?a * 1 → ?a.
 */
public final class IdentityOne extends RewriteRulePrimitive {
    public IdentityOne() {
        super("identity-one",
                Pattern.mul(Pattern.hole("a"), Pattern.lit(1)),
                Pattern.hole("a"));
    }
}
