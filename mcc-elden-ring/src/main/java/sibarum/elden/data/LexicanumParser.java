package sibarum.elden.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Parses the Lexicanum RAG corpus format. Each entry has this shape:
 *
 *   --- ENTRY START ---
 *   "title":"Grand Cathay",
 *   "type":"Faction",
 *   "aliases":["Grand Empire of Cathay", "Imperial Cathan", ...],
 *   "features":{...},
 *   "relationships":{...}
 *   ---
 *   Free-text prose paragraph(s) until the next entry or EOF.
 *
 * We only extract {@code title}, {@code type}, {@code aliases}, and the prose
 * body — that's enough to bootstrap NER labels. Features and relationships
 * are read but not retained here (a future relation extractor can re-parse).
 *
 * The header values are JSON-like but not strict JSON, so this is a small
 * line-by-line parser rather than a JSON library call.
 */
public final class LexicanumParser {

    public record Entry(String title, String type, List<String> aliases, String prose) {}

    private static final String ENTRY_START = "--- ENTRY START ---";
    private static final String SEPARATOR = "---";

    private LexicanumParser() {}

    public static List<Entry> parse(Path file) throws IOException {
        List<Entry> entries = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(file)) {
            String title = null;
            String type = null;
            List<String> aliases = new ArrayList<>();
            StringBuilder prose = new StringBuilder();
            boolean inHeader = false;
            boolean inProse = false;

            String line;
            while ((line = r.readLine()) != null) {
                String trimmed = line.strip();
                if (trimmed.equals(ENTRY_START)) {
                    if (title != null) {
                        entries.add(new Entry(title, type == null ? "" : type,
                                List.copyOf(aliases), prose.toString().strip()));
                    }
                    title = null; type = null;
                    aliases.clear();
                    prose.setLength(0);
                    inHeader = true;
                    inProse = false;
                } else if (trimmed.equals(SEPARATOR) && inHeader) {
                    inHeader = false;
                    inProse = true;
                } else if (inHeader) {
                    String key = headerKey(trimmed);
                    if (key == null) continue;
                    switch (key) {
                        case "title" -> { Optional<String> v = headerStringValue(trimmed); v.ifPresent(s -> { /* set below */ }); title = v.orElse(null); }
                        case "type"  -> type = headerStringValue(trimmed).orElse(null);
                        case "aliases" -> aliases.addAll(headerStringList(trimmed));
                        default -> {}
                    }
                } else if (inProse) {
                    if (!trimmed.isEmpty()) {
                        if (prose.length() > 0) prose.append(' ');
                        prose.append(trimmed);
                    }
                }
            }
            if (title != null) {
                entries.add(new Entry(title, type == null ? "" : type,
                        List.copyOf(aliases), prose.toString().strip()));
            }
        }
        return entries;
    }

    private static String headerKey(String line) {
        if (!line.startsWith("\"")) return null;
        int closeQuote = line.indexOf('"', 1);
        if (closeQuote < 0) return null;
        return line.substring(1, closeQuote);
    }

    /** Returns the string value from a line like {@code "key":"value",} */
    private static Optional<String> headerStringValue(String line) {
        int colon = line.indexOf(':');
        if (colon < 0) return Optional.empty();
        String rhs = line.substring(colon + 1).strip();
        if (rhs.endsWith(",")) rhs = rhs.substring(0, rhs.length() - 1).strip();
        if (rhs.startsWith("\"") && rhs.endsWith("\"") && rhs.length() >= 2) {
            return Optional.of(rhs.substring(1, rhs.length() - 1));
        }
        return Optional.empty();
    }

    /** Returns the string list from a line like {@code "key":["a", "b"],} */
    private static List<String> headerStringList(String line) {
        int open = line.indexOf('[');
        int close = line.lastIndexOf(']');
        if (open < 0 || close <= open) return List.of();
        String inner = line.substring(open + 1, close);
        List<String> out = new ArrayList<>();
        // Split on quoted strings.
        int i = 0;
        while (i < inner.length()) {
            int q1 = inner.indexOf('"', i);
            if (q1 < 0) break;
            int q2 = inner.indexOf('"', q1 + 1);
            if (q2 < 0) break;
            String s = inner.substring(q1 + 1, q2);
            if (!s.isEmpty()) out.add(s);
            i = q2 + 1;
        }
        return out;
    }
}
