package sibarum.strnn.primitive;

import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.List;

/**
 * Tokenize natural-language text into word and punctuation tokens.
 *
 * A token is one of:
 *   - a contiguous run of letters, digits, apostrophes, or hyphens (a "word"),
 *   - a single punctuation character (comma, period, colon, semicolon,
 *     em-dash, parenthesis, etc.) emitted as its own one-character token.
 *
 * Whitespace separates tokens and is not itself emitted. Case is preserved so
 * proper nouns (e.g. "Marika") remain distinct from common words.
 */
public final class TextTokenize implements Primitive {

    @Override
    public String name() {
        return "text-tokenize";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TOKEN_LIST;
    }

    @Override
    public Value apply(List<Value> inputs) {
        String text = ((StringValue) inputs.getFirst()).s();
        List<String> tokens = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (isWordChar(c)) {
                cur.append(c);
            } else {
                flush(cur, tokens);
                if (!Character.isWhitespace(c)) {
                    tokens.add(String.valueOf(c));
                }
            }
        }
        flush(cur, tokens);
        return new TokenListValue(tokens);
    }

    private static boolean isWordChar(char c) {
        return Character.isLetterOrDigit(c) || c == '\'' || c == '-';
    }

    private static void flush(StringBuilder cur, List<String> tokens) {
        if (cur.length() > 0) {
            tokens.add(cur.toString());
            cur.setLength(0);
        }
    }
}
