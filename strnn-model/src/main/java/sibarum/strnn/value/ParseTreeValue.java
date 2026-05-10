package sibarum.strnn.value;

import java.util.Objects;

/**
 * Tree-shaped value: an arithmetic parse tree with literal leaves,
 * variable leaves, and binary-operator interior nodes. Equality is
 * structural by virtue of being records; ValueDistance treats trees as
 * exact-match (1.0 if unequal, 0.0 if equal).
 *
 * This is the §2.4 &quot;values as graphs&quot; principle made operational —
 * symbolic rewrite primitives in v2 transform ParseTreeValue → ParseTreeValue.
 */
public sealed interface ParseTreeValue extends Value
        permits ParseTreeValue.Literal,
                ParseTreeValue.Variable,
                ParseTreeValue.BinaryOp,
                ParseTreeValue.Omega {

    @Override
    default ValueType type() {
        return ValueType.PARSE_TREE;
    }

    record Literal(double value) implements ParseTreeValue {
        @Override
        public String toString() {
            if (value == Math.floor(value) && !Double.isInfinite(value)) {
                return Integer.toString((int) value);
            }
            return Double.toString(value);
        }
    }

    /**
     * The reciprocal of zero in the total, reversible algebra. Distinct from
     * any Literal: ω is not a number you can compute with ordinary IEEE doubles
     * (which would give NaN for 0·ω). Rewrite rules give it its meaning —
     * notably 0·ω → 1 in the wider rule library — and its presence as a
     * dedicated variant prevents accidental coercion to Double.POSITIVE_INFINITY.
     */
    record Omega() implements ParseTreeValue {
        public static final Omega INSTANCE = new Omega();

        @Override
        public String toString() {
            return "ω";
        }
    }

    record Variable(String name) implements ParseTreeValue {
        public Variable {
            Objects.requireNonNull(name);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    record BinaryOp(Operator op, ParseTreeValue left, ParseTreeValue right) implements ParseTreeValue {
        public BinaryOp {
            Objects.requireNonNull(op);
            Objects.requireNonNull(left);
            Objects.requireNonNull(right);
        }

        @Override
        public String toString() {
            return "(" + left + " " + op.symbol() + " " + right + ")";
        }
    }

    /** Convenience factories that read more naturally inline. */
    static ParseTreeValue lit(double v) {
        return new Literal(v);
    }

    static ParseTreeValue var(String name) {
        return new Variable(name);
    }

    static ParseTreeValue add(ParseTreeValue l, ParseTreeValue r) {
        return new BinaryOp(Operator.ADD, l, r);
    }

    static ParseTreeValue sub(ParseTreeValue l, ParseTreeValue r) {
        return new BinaryOp(Operator.SUB, l, r);
    }

    static ParseTreeValue mul(ParseTreeValue l, ParseTreeValue r) {
        return new BinaryOp(Operator.MUL, l, r);
    }

    static ParseTreeValue div(ParseTreeValue l, ParseTreeValue r) {
        return new BinaryOp(Operator.DIV, l, r);
    }

    static ParseTreeValue omega() {
        return Omega.INSTANCE;
    }

    /** Depth-first count of nodes in the tree. */
    default int size() {
        return switch (this) {
            case Literal l -> 1;
            case Variable v -> 1;
            case Omega o -> 1;
            case BinaryOp b -> 1 + b.left.size() + b.right.size();
        };
    }
}
