package sibarum.strnn.rewrite;

/**
 * Symbolic rewrite rule: (?a + ?b) * ?c → ?a*?c + ?b*?c.
 *
 * Pure symbolic; produces a structurally larger tree. Useful when downstream
 * rules can simplify the distributed form. v2 includes this primitive but
 * does not exercise it in the two-variant demo (we'd need sub-tree rule
 * application — deferred to v3).
 */
public final class Distribute extends RewriteRulePrimitive {
    public Distribute() {
        super("distribute",
                Pattern.mul(
                        Pattern.add(Pattern.hole("a"), Pattern.hole("b")),
                        Pattern.hole("c")),
                Pattern.add(
                        Pattern.mul(Pattern.hole("a"), Pattern.hole("c")),
                        Pattern.mul(Pattern.hole("b"), Pattern.hole("c"))));
    }
}
