package sibarum.strnn.rewrite;

import sibarum.strnn.value.ParseTreeValue;

import java.util.Optional;

/**
 * Pattern-matching and substitution over ParseTreeValue.
 *
 * Matching: returns an Optional&lt;Bindings&gt; — empty means the pattern did
 * not match the tree. A pattern with non-linear holes (repeated names) is
 * matched by binding on first occurrence and verifying structural equality
 * on subsequent occurrences.
 *
 * Substitution: requires every hole referenced by the pattern to be bound;
 * throws if a hole is missing.
 */
public final class Matcher {
    private Matcher() {
    }

    public static Optional<Bindings> match(Pattern p, ParseTreeValue t) {
        Bindings.Builder b = new Bindings.Builder();
        if (matchInto(p, t, b)) return Optional.of(b.build());
        return Optional.empty();
    }

    private static boolean matchInto(Pattern p, ParseTreeValue t, Bindings.Builder b) {
        return switch (p) {
            case Pattern.LitPat lp -> t instanceof ParseTreeValue.Literal l && l.value() == lp.value();
            case Pattern.VarPat vp -> t instanceof ParseTreeValue.Variable v && v.name().equals(vp.name());
            case Pattern.HolePat hp -> b.bind(hp.name(), t);
            case Pattern.OpPat op -> t instanceof ParseTreeValue.BinaryOp bo
                    && bo.op() == op.op()
                    && matchInto(op.left(), bo.left(), b)
                    && matchInto(op.right(), bo.right(), b);
        };
    }

    public static ParseTreeValue substitute(Pattern p, Bindings b) {
        return switch (p) {
            case Pattern.LitPat lp -> new ParseTreeValue.Literal(lp.value());
            case Pattern.VarPat vp -> new ParseTreeValue.Variable(vp.name());
            case Pattern.HolePat hp -> b.require(hp.name());
            case Pattern.OpPat op -> new ParseTreeValue.BinaryOp(
                    op.op(),
                    substitute(op.left(), b),
                    substitute(op.right(), b));
        };
    }
}
