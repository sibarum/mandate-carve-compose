package sibarum.strnn.primitive;

import sibarum.strnn.value.Operator;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Recursive-descent parser for arithmetic expressions over single-digit
 * literals and single-letter variables, with operators + - * / and parens.
 *
 * Grammar:
 *   expr   = term (('+' | '-') term)*
 *   term   = factor (('*' | '/') factor)*
 *   factor = NUMBER | VARIABLE | '(' expr ')'
 *
 * Whitespace is skipped. NUMBER is one or more digits with optional decimal
 * point. VARIABLE is a single ASCII letter.
 */
public final class ParseExpression implements Primitive {
    @Override
    public String name() {
        return "parse-expression";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.PARSE_TREE;
    }

    @Override
    public Value apply(List<Value> inputs) {
        StringValue s = (StringValue) inputs.getFirst();
        Cursor c = new Cursor(s.s());
        ParseTreeValue tree = parseExpr(c);
        c.skipWhitespace();
        if (!c.atEnd()) {
            throw new IllegalArgumentException(
                    "unexpected trailing characters at index " + c.pos + " in: " + s.s());
        }
        return tree;
    }

    private static ParseTreeValue parseExpr(Cursor c) {
        ParseTreeValue left = parseTerm(c);
        while (true) {
            c.skipWhitespace();
            if (c.atEnd()) break;
            char ch = c.peek();
            if (ch != '+' && ch != '-') break;
            c.advance();
            ParseTreeValue right = parseTerm(c);
            left = new ParseTreeValue.BinaryOp(Operator.fromChar(ch), left, right);
        }
        return left;
    }

    private static ParseTreeValue parseTerm(Cursor c) {
        ParseTreeValue left = parseFactor(c);
        while (true) {
            c.skipWhitespace();
            if (c.atEnd()) break;
            char ch = c.peek();
            if (ch != '*' && ch != '/') break;
            c.advance();
            ParseTreeValue right = parseFactor(c);
            left = new ParseTreeValue.BinaryOp(Operator.fromChar(ch), left, right);
        }
        return left;
    }

    private static ParseTreeValue parseFactor(Cursor c) {
        c.skipWhitespace();
        if (c.atEnd()) throw new IllegalArgumentException("unexpected end of expression");
        char ch = c.peek();
        if (ch == '(') {
            c.advance();
            ParseTreeValue inner = parseExpr(c);
            c.skipWhitespace();
            if (c.atEnd() || c.peek() != ')') {
                throw new IllegalArgumentException("expected ')' at index " + c.pos);
            }
            c.advance();
            return inner;
        }
        if (Character.isDigit(ch) || ch == '.') {
            int start = c.pos;
            boolean sawDot = false;
            while (!c.atEnd()) {
                char d = c.peek();
                if (Character.isDigit(d)) {
                    c.advance();
                } else if (d == '.' && !sawDot) {
                    sawDot = true;
                    c.advance();
                } else {
                    break;
                }
            }
            return new ParseTreeValue.Literal(Double.parseDouble(c.s.substring(start, c.pos)));
        }
        if (Character.isLetter(ch)) {
            c.advance();
            return new ParseTreeValue.Variable(String.valueOf(ch));
        }
        throw new IllegalArgumentException("unexpected character '" + ch + "' at index " + c.pos);
    }

    private static final class Cursor {
        final String s;
        int pos = 0;

        Cursor(String s) {
            this.s = s;
        }

        boolean atEnd() {
            return pos >= s.length();
        }

        char peek() {
            return s.charAt(pos);
        }

        void advance() {
            pos++;
        }

        void skipWhitespace() {
            while (!atEnd() && Character.isWhitespace(s.charAt(pos))) pos++;
        }
    }
}
