package sibarum.elden.pos;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal CoNLL-U format parser. Reads a {@code .conllu} file (such as the
 * Universal Dependencies English EWT train file) and returns sentences as
 * lists of (surface form, universal POS) pairs.
 *
 * Each non-blank, non-comment line is one token, tab-separated. We use:
 *   - column 1 (1-based ID) — to detect multiword/empty tokens to skip
 *   - column 2 — FORM (surface form)
 *   - column 4 — UPOS (universal part-of-speech tag)
 *
 * Multiword tokens (IDs containing '-') and empty nodes (containing '.') are
 * skipped — their children carry the actual annotation.
 *
 * Sentences are separated by blank lines.
 */
public final class ConlluParser {

    public record TaggedToken(String form, String upos) {}
    public record Sentence(List<TaggedToken> tokens) {}

    private ConlluParser() {}

    public static List<Sentence> parse(Path file) throws IOException {
        List<Sentence> sentences = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(file)) {
            List<TaggedToken> current = new ArrayList<>();
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!current.isEmpty()) {
                        sentences.add(new Sentence(List.copyOf(current)));
                        current.clear();
                    }
                    continue;
                }
                if (line.startsWith("#")) continue;

                String[] cols = line.split("\t");
                if (cols.length < 5) continue;

                String id = cols[0];
                if (id.contains("-") || id.contains(".")) continue;

                String form = cols[1];
                String upos = cols[3];
                if (form.isEmpty() || upos.isEmpty() || upos.equals("_")) continue;

                current.add(new TaggedToken(form, upos));
            }
            if (!current.isEmpty()) {
                sentences.add(new Sentence(List.copyOf(current)));
            }
        }
        return sentences;
    }
}
