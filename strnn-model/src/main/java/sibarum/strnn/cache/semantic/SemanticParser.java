package sibarum.strnn.cache.semantic;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Recursive-descent parser for the sample-semantics file format.
 *
 * Grammar:
 * <pre>
 *   line       = dichotomy "~>" expr
 *   dichotomy  = "(" IDENT "|" IDENT ")"
 *   expr       = union
 *   union      = compose ("&" compose)*           // flattened into Union
 *   compose    = qualified (("+" | "*") qualified)*  // left-assoc binary
 *   qualified  = atomic (":" atomic)*              // left-assoc
 *   atomic     = IDENT | "(" expr ")"
 * </pre>
 *
 * Comments ({@code # ...}) and blank lines are skipped. Comma is tolerated
 * as a synonym for {@code &} (one line in the source uses comma where the
 * surrounding lines use {@code &}; treat as an alternate spelling rather than
 * a syntax error).
 */
public final class SemanticParser {

    private SemanticParser() {
    }

    /** Parse every non-comment, non-blank line in {@code source} as a relation. */
    public static List<SemRelation> parseAll(String source) {
        List<SemRelation> out = new ArrayList<>();
        String[] lines = source.split("\\R");
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].strip();
            if (line.isEmpty() || line.startsWith("#")) continue;
            try {
                out.add(parseLine(line));
            } catch (RuntimeException e) {
                throw new IllegalArgumentException(
                        "parse error on line " + (i + 1) + ": '" + lines[i] + "'", e);
            }
        }
        return out;
    }

    /** Parse a single relation. Throws if the line has trailing tokens. */
    public static SemRelation parseLine(String line) {
        Parser p = new Parser(new Lexer(line));
        SemRelation r = p.parseRelation();
        p.expectEof();
        return r;
    }

    /** Collect every atom appearing anywhere across the relation list. */
    public static Set<String> collectAtoms(List<SemRelation> relations) {
        Set<String> atoms = new LinkedHashSet<>();
        for (SemRelation r : relations) {
            atoms.add(r.lhs().left());
            atoms.add(r.lhs().right());
            collectAtoms(r.rhs(), atoms);
        }
        return atoms;
    }

    private static void collectAtoms(SemExpr e, Set<String> out) {
        switch (e) {
            case SemExpr.Atom a -> out.add(a.name());
            case SemExpr.Qualified q -> {
                collectAtoms(q.head(), out);
                collectAtoms(q.facet(), out);
            }
            case SemExpr.Composition c -> {
                collectAtoms(c.left(), out);
                collectAtoms(c.right(), out);
            }
            case SemExpr.Conjunction c -> {
                collectAtoms(c.left(), out);
                collectAtoms(c.right(), out);
            }
            case SemExpr.Union u -> {
                for (SemExpr m : u.members()) collectAtoms(m, out);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Lexer
    // -----------------------------------------------------------------------

    enum TokenType {
        LPAREN, RPAREN, PIPE, ARROW, COLON, AMP, STAR, PLUS, IDENT, EOF
    }

    record Token(TokenType type, String text, int pos) {
        @Override
        public String toString() {
            return type + "('" + text + "'@" + pos + ")";
        }
    }

    static final class Lexer {
        private final String src;
        private int pos;

        Lexer(String src) {
            this.src = src;
            this.pos = 0;
        }

        Token next() {
            while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
                pos++;
            }
            if (pos >= src.length()) return new Token(TokenType.EOF, "", pos);

            char c = src.charAt(pos);
            int start = pos;
            switch (c) {
                case '(' -> { pos++; return new Token(TokenType.LPAREN, "(", start); }
                case ')' -> { pos++; return new Token(TokenType.RPAREN, ")", start); }
                case '|' -> { pos++; return new Token(TokenType.PIPE, "|", start); }
                case ':' -> { pos++; return new Token(TokenType.COLON, ":", start); }
                case '&' -> { pos++; return new Token(TokenType.AMP, "&", start); }
                case '*' -> { pos++; return new Token(TokenType.STAR, "*", start); }
                case '+' -> { pos++; return new Token(TokenType.PLUS, "+", start); }
                case ',' -> { pos++; return new Token(TokenType.AMP, "&", start); } // comma → &
                case '~' -> {
                    if (pos + 1 < src.length() && src.charAt(pos + 1) == '>') {
                        pos += 2;
                        return new Token(TokenType.ARROW, "~>", start);
                    }
                    throw new IllegalArgumentException(
                            "expected '~>' at position " + pos + ", got '~" + (pos + 1 < src.length() ? src.charAt(pos + 1) : "") + "'");
                }
                default -> {
                    if (Character.isLetter(c) || c == '_') {
                        while (pos < src.length()
                                && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) {
                            pos++;
                        }
                        return new Token(TokenType.IDENT, src.substring(start, pos), start);
                    }
                    throw new IllegalArgumentException(
                            "unexpected character '" + c + "' at position " + pos);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Parser
    // -----------------------------------------------------------------------

    static final class Parser {
        private final Lexer lex;
        private Token current;

        Parser(Lexer lex) {
            this.lex = lex;
            this.current = lex.next();
        }

        SemRelation parseRelation() {
            Dichotomy d = parseDichotomy();
            expect(TokenType.ARROW);
            SemExpr rhs = parseExpr();
            return new SemRelation(d, rhs);
        }

        Dichotomy parseDichotomy() {
            expect(TokenType.LPAREN);
            String left = expectIdent();
            expect(TokenType.PIPE);
            String right = expectIdent();
            expect(TokenType.RPAREN);
            return new Dichotomy(left, right);
        }

        SemExpr parseExpr() {
            return parseUnion();
        }

        SemExpr parseUnion() {
            SemExpr first = parseCompose();
            if (current.type != TokenType.AMP) return first;
            List<SemExpr> members = new ArrayList<>();
            members.add(first);
            while (current.type == TokenType.AMP) {
                advance();
                members.add(parseCompose());
            }
            return new SemExpr.Union(members);
        }

        SemExpr parseCompose() {
            SemExpr left = parseQualified();
            while (current.type == TokenType.STAR || current.type == TokenType.PLUS) {
                TokenType op = current.type;
                advance();
                SemExpr right = parseQualified();
                left = (op == TokenType.STAR)
                        ? new SemExpr.Composition(left, right)
                        : new SemExpr.Conjunction(left, right);
            }
            return left;
        }

        SemExpr parseQualified() {
            SemExpr head = parseAtomic();
            while (current.type == TokenType.COLON) {
                advance();
                SemExpr facet = parseAtomic();
                head = new SemExpr.Qualified(head, facet);
            }
            return head;
        }

        SemExpr parseAtomic() {
            if (current.type == TokenType.IDENT) {
                String name = current.text;
                advance();
                return new SemExpr.Atom(name);
            }
            if (current.type == TokenType.LPAREN) {
                advance();
                SemExpr inner = parseExpr();
                expect(TokenType.RPAREN);
                return inner;
            }
            throw new IllegalStateException(
                    "expected identifier or '(' but got " + current);
        }

        void advance() {
            current = lex.next();
        }

        void expect(TokenType t) {
            if (current.type != t) {
                throw new IllegalStateException("expected " + t + " but got " + current);
            }
            advance();
        }

        String expectIdent() {
            if (current.type != TokenType.IDENT) {
                throw new IllegalStateException("expected identifier but got " + current);
            }
            String s = current.text;
            advance();
            return s;
        }

        void expectEof() {
            if (current.type != TokenType.EOF) {
                throw new IllegalStateException("expected end of line but got " + current);
            }
        }
    }
}
