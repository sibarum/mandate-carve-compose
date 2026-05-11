package sibarum.elden.embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * Same tokenization rules as the framework's {@code TextTokenize}, but also
 * emits the (startChar, endChar) range each token came from in the original
 * source string. Used at training time to align EntitySpan character offsets
 * with token positions.
 *
 * A token is one of:
 *   - a contiguous run of letters, digits, apostrophes, or hyphens (a word),
 *   - a single punctuation character (one-character token).
 * Whitespace separates tokens and is not itself emitted.
 *
 * For each emitted token, {@code endChar} is exclusive (Java substring style).
 */
public final class OffsetTokenizer {

    public record OffsetToken(String text, int startChar, int endChar) {}

    private OffsetTokenizer() {}

    public static List<OffsetToken> tokenize(String text) {
        List<OffsetToken> tokens = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        int tokenStart = -1;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (isWordChar(c)) {
                if (cur.isEmpty()) tokenStart = i;
                cur.append(c);
            } else {
                flush(cur, tokenStart, i, tokens);
                cur.setLength(0);
                tokenStart = -1;
                if (!Character.isWhitespace(c)) {
                    tokens.add(new OffsetToken(String.valueOf(c), i, i + 1));
                }
            }
        }
        flush(cur, tokenStart, text.length(), tokens);
        return tokens;
    }

    private static boolean isWordChar(char c) {
        return Character.isLetterOrDigit(c) || c == '\'' || c == '-';
    }

    private static void flush(StringBuilder cur, int tokenStart, int endExclusive, List<OffsetToken> tokens) {
        if (!cur.isEmpty()) {
            tokens.add(new OffsetToken(cur.toString(), tokenStart, endExclusive));
        }
    }
}
