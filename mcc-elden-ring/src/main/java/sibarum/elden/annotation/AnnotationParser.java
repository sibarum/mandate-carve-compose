package sibarum.elden.annotation;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Parses inline-markup annotation text into an AnnotatedItem.
 *
 * Syntax:
 *   [surface form](type:entity_id)
 *
 * Example:
 *   "The [crown](artifact:dawn_crown) of [Eldred](character:eldred)."
 *
 * Markup is stripped to produce the raw text, and each `[...]` becomes an EntitySpan
 * with start/end offsets into the raw text. The type token must match an EntityType
 * value (case-insensitive). Entity IDs are free-form strings.
 */
public final class AnnotationParser {

    private AnnotationParser() {}

    public static AnnotatedItem parse(String itemId, String markedText, List<Relation> relations) {
        StringBuilder raw = new StringBuilder();
        List<EntitySpan> spans = new ArrayList<>();
        int i = 0;
        while (i < markedText.length()) {
            char c = markedText.charAt(i);
            if (c == '[') {
                int endBracket = markedText.indexOf(']', i);
                if (endBracket == -1) {
                    throw new IllegalArgumentException("unclosed '[' at offset " + i + " in item " + itemId);
                }
                if (endBracket + 1 >= markedText.length() || markedText.charAt(endBracket + 1) != '(') {
                    throw new IllegalArgumentException("expected '(' after ']' at offset " + endBracket + " in item " + itemId);
                }
                int endParen = markedText.indexOf(')', endBracket + 1);
                if (endParen == -1) {
                    throw new IllegalArgumentException("unclosed '(' at offset " + (endBracket + 1) + " in item " + itemId);
                }
                String surface = markedText.substring(i + 1, endBracket);
                String typeAndId = markedText.substring(endBracket + 2, endParen);
                int colon = typeAndId.indexOf(':');
                if (colon == -1) {
                    throw new IllegalArgumentException("expected 'type:id' in (" + typeAndId + ") in item " + itemId);
                }
                String typeToken = typeAndId.substring(0, colon).trim().toUpperCase(Locale.ROOT);
                String entityId = typeAndId.substring(colon + 1).trim();
                EntityType type;
                try {
                    type = EntityType.valueOf(typeToken);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("unknown entity type '" + typeToken + "' in item " + itemId);
                }
                int start = raw.length();
                raw.append(surface);
                int end = raw.length();
                spans.add(new EntitySpan(start, end, surface, type, entityId));
                i = endParen + 1;
            } else {
                raw.append(c);
                i++;
            }
        }
        return new AnnotatedItem(itemId, raw.toString(), List.copyOf(spans), List.copyOf(relations));
    }
}
