package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: ?a*?c + ?b*?c → (?a + ?b) * ?c.
 *
 * Inverse of Distribute. Non-linear in ?c — only matches when both
 * multiplications share the same right operand. Same v2 limitation as
 * Distribute: included in the library but not exercised by the two-variant
 * demo.
 */
public final class FactorCommon extends RewriteRulePrimitive {
    public FactorCommon() {
        super("factor-common",
                Pattern.add(
                        Pattern.mul(Pattern.hole("a"), Pattern.hole("c")),
                        Pattern.mul(Pattern.hole("b"), Pattern.hole("c"))),
                Pattern.mul(
                        Pattern.add(Pattern.hole("a"), Pattern.hole("b")),
                        Pattern.hole("c")));
    }
}
