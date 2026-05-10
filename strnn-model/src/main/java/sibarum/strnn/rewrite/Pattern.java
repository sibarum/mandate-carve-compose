package sibarum.strnn.rewrite;

import sibarum.strnn.value.Operator;

/**
 * A tree-shaped pattern over ParseTreeValue. Used both as a matcher (against
 * a concrete tree) and as a template (for substitution). Holes are named
 * placeholders that bind during matching and are replaced during substitution.
 *
 * Linear patterns (each hole used once) are simple to match. Non-linear
 * patterns (a hole repeats, e.g. {@code ?a + ?a}) require the matcher to
 * verify the second occurrence equals the first binding; supported by the
 * Matcher.
 */
public sealed interface Pattern {

    record LitPat(double value) implements Pattern {
        @Override
        public String toString() {
            if (value == Math.floor(value) && !Double.isInfinite(value)) {
                return Integer.toString((int) value);
            }
            return Double.toString(value);
        }
    }

    record VarPat(String name) implements Pattern {
        @Override
        public String toString() {
            return name;
        }
    }

    record HolePat(String name) implements Pattern {
        @Override
        public String toString() {
            return "?" + name;
        }
    }

    record OpPat(Operator op, Pattern left, Pattern right) implements Pattern {
        @Override
        public String toString() {
            return "(" + left + " " + op.symbol() + " " + right + ")";
        }
    }

    static Pattern lit(double v) {
        return new LitPat(v);
    }

    static Pattern var(String name) {
        return new VarPat(name);
    }

    static Pattern hole(String name) {
        return new HolePat(name);
    }

    static Pattern add(Pattern l, Pattern r) {
        return new OpPat(Operator.ADD, l, r);
    }

    static Pattern sub(Pattern l, Pattern r) {
        return new OpPat(Operator.SUB, l, r);
    }

    static Pattern mul(Pattern l, Pattern r) {
        return new OpPat(Operator.MUL, l, r);
    }

    static Pattern div(Pattern l, Pattern r) {
        return new OpPat(Operator.DIV, l, r);
    }
}
