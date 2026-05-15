package sibarum.mcc.serialization;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal reflection-free JSON parser. Produces a generic tree of
 * {@code null}, {@link Boolean}, {@link Long} (for integral numbers),
 * {@link Double} (for fractional numbers), {@link String},
 * {@link List} (always {@code ArrayList}), and
 * {@link Map} (always {@code LinkedHashMap}). Object key order is
 * preserved.
 *
 * <p>Reflection-free; Graal native-image friendly. Strict-ish:
 * accepts JSON whitespace but no trailing commas or unquoted keys.
 *
 * <p>Numeric parsing: any number with a {@code .}, {@code e}, or
 * {@code E} is decoded as {@code Double}; otherwise as {@code Long}.
 * Callers that need ints must downcast (the schema record fields
 * declare the expected type).
 */
public final class JsonReader {

    private final String src;
    private int pos;

    public JsonReader(String src) {
        this.src = src;
        this.pos = 0;
    }

    /** Parse one JSON value from {@code src}; rejects trailing non-whitespace. */
    public static Object parse(String src) {
        JsonReader r = new JsonReader(src);
        r.skipWs();
        Object v = r.readValue();
        r.skipWs();
        if (r.pos != r.src.length()) {
            throw new IllegalArgumentException(
                    "JsonReader: trailing content at position " + r.pos);
        }
        return v;
    }

    private Object readValue() {
        skipWs();
        if (pos >= src.length()) throw err("unexpected EOF");
        char c = src.charAt(pos);
        return switch (c) {
            case '{' -> readObject();
            case '[' -> readArray();
            case '"' -> readString();
            case 't', 'f' -> readBoolean();
            case 'n' -> readNull();
            default -> readNumber();
        };
    }

    private Map<String, Object> readObject() {
        expect('{');
        Map<String, Object> m = new LinkedHashMap<>();
        skipWs();
        if (peek() == '}') {
            pos++;
            return m;
        }
        while (true) {
            skipWs();
            String key = readString();
            skipWs();
            expect(':');
            Object value = readValue();
            m.put(key, value);
            skipWs();
            char next = expectOneOf(',', '}');
            if (next == '}') return m;
        }
    }

    private List<Object> readArray() {
        expect('[');
        List<Object> list = new ArrayList<>();
        skipWs();
        if (peek() == ']') {
            pos++;
            return list;
        }
        while (true) {
            list.add(readValue());
            skipWs();
            char next = expectOneOf(',', ']');
            if (next == ']') return list;
        }
    }

    private String readString() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= src.length()) throw err("unterminated string");
            char c = src.charAt(pos++);
            if (c == '"') return sb.toString();
            if (c == '\\') {
                if (pos >= src.length()) throw err("dangling backslash in string");
                char esc = src.charAt(pos++);
                switch (esc) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (pos + 4 > src.length()) throw err("bad \\u escape");
                        int code = Integer.parseInt(src.substring(pos, pos + 4), 16);
                        sb.append((char) code);
                        pos += 4;
                    }
                    default -> throw err("invalid escape \\" + esc);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Boolean readBoolean() {
        if (src.startsWith("true", pos)) { pos += 4; return Boolean.TRUE; }
        if (src.startsWith("false", pos)) { pos += 5; return Boolean.FALSE; }
        throw err("expected boolean");
    }

    private Object readNull() {
        if (src.startsWith("null", pos)) { pos += 4; return null; }
        throw err("expected null");
    }

    private Object readNumber() {
        int start = pos;
        if (peek() == '-') pos++;
        while (pos < src.length() && isNumberChar(src.charAt(pos))) pos++;
        String token = src.substring(start, pos);
        if (token.isEmpty() || token.equals("-")) throw err("invalid number");
        boolean fractional = token.contains(".") || token.contains("e") || token.contains("E");
        // NOTE: avoid `fractional ? Double.valueOf(token) : Long.valueOf(token)` —
        // Java's ternary numeric promotion would silently unbox Long to double
        // and reBox as Double, losing the integer / fractional distinction.
        try {
            if (fractional) {
                return Double.valueOf(token);
            }
            return Long.valueOf(token);
        } catch (NumberFormatException e) {
            throw err("invalid number: " + token);
        }
    }

    private static boolean isNumberChar(char c) {
        return (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
    }

    private void skipWs() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos++;
            } else {
                return;
            }
        }
    }

    private char peek() {
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private void expect(char c) {
        if (pos >= src.length() || src.charAt(pos) != c) throw err("expected '" + c + "'");
        pos++;
    }

    private char expectOneOf(char a, char b) {
        if (pos >= src.length()) throw err("expected '" + a + "' or '" + b + "'");
        char c = src.charAt(pos);
        if (c != a && c != b) throw err("expected '" + a + "' or '" + b + "' but found '" + c + "'");
        pos++;
        return c;
    }

    private IllegalArgumentException err(String msg) {
        int near = Math.min(src.length() - pos, 20);
        return new IllegalArgumentException(
                "JsonReader at position " + pos + ": " + msg
                        + " near \"" + src.substring(pos, pos + near) + "\"");
    }
}
