package sibarum.elden.data;

import sibarum.elden.data.ParquetBioLoader.Sentence;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * Loads BIO-labeled training sentences from a newline-delimited JSON file with
 * one Q&amp;A record per line. Each row's {@code output} field is the prose
 * answer, and {@code metadata.entity_name} is the canonical entity surface
 * form known to be mentioned in that prose.
 *
 * For each row, this loader:
 * <ol>
 *   <li>Tokenizes the {@code output} text with {@link OffsetTokenizer}.</li>
 *   <li>Searches for every occurrence of {@code entity_name} as a contiguous
 *       case-insensitive token-aligned subsequence within the tokenized
 *       output.</li>
 *   <li>Labels matching tokens B (first match token) / I (subsequent match
 *       tokens) and everything else O.</li>
 *   <li>Drops rows with no successful match (a small fraction — typically
 *       the answer paraphrases the entity).</li>
 * </ol>
 *
 * Read via DuckDB's native {@code read_json_auto} (no JSON library on the
 * classpath).
 */
public final class EldenJsonlBioLoader {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;

    public record LoadStats(int totalRows, int matchedRows, int skippedRows,
                             long totalTokens, long entityTokens, int totalSpans) {}

    public static final class Result {
        public final List<Sentence> sentences;
        public final LoadStats stats;
        Result(List<Sentence> sentences, LoadStats stats) {
            this.sentences = sentences;
            this.stats = stats;
        }
    }

    private EldenJsonlBioLoader() {}

    public static Result load(Path file, OptionalInt limit) throws SQLException, ClassNotFoundException {
        Class.forName("org.duckdb.DuckDBDriver");
        String src = "read_json_auto('" + file.toString().replace("'", "''") + "', format='nd')";
        String sql = "SELECT output, metadata.entity_name AS entity_name FROM " + src
                + " WHERE output IS NOT NULL AND metadata.entity_name IS NOT NULL"
                + limit.stream().mapToObj(n -> " LIMIT " + n).findFirst().orElse("");

        List<Sentence> out = new ArrayList<>();
        int totalRows = 0, matchedRows = 0, skipped = 0, totalSpans = 0;
        long totalTokens = 0, entityTokens = 0;

        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                totalRows++;
                String output = rs.getString(1);
                String entity = rs.getString(2);
                if (output == null || output.isBlank() || entity == null || entity.isBlank()) {
                    skipped++; continue;
                }
                entity = entity.trim();

                List<OffsetToken> outputToks = OffsetTokenizer.tokenize(output);
                if (outputToks.isEmpty()) { skipped++; continue; }
                List<OffsetToken> entityToks = OffsetTokenizer.tokenize(entity);
                if (entityToks.isEmpty()) { skipped++; continue; }

                List<String> forms = new ArrayList<>(outputToks.size());
                for (OffsetToken t : outputToks) forms.add(t.text());

                int[] bio = new int[forms.size()];
                int matches = labelMatches(forms, entityToks, bio);
                if (matches == 0) { skipped++; continue; }

                matchedRows++;
                totalSpans += matches;
                totalTokens += forms.size();
                for (int b : bio) if (b != LABEL_O) entityTokens++;
                out.add(new Sentence(forms, bio));
            }
        }

        LoadStats stats = new LoadStats(totalRows, matchedRows, skipped,
                totalTokens, entityTokens, totalSpans);
        System.out.printf("EldenJsonlBioLoader: %,d rows -> %,d labeled (%,d skipped)%n",
                totalRows, matchedRows, skipped);
        System.out.printf("  %,d tokens, %,d entity (%.1f%%), %,d spans%n",
                totalTokens, entityTokens,
                100.0 * entityTokens / Math.max(totalTokens, 1),
                totalSpans);
        return new Result(out, stats);
    }

    /**
     * Find every contiguous case-insensitive token-aligned occurrence of
     * {@code entityToks} inside {@code forms} and label the BIO array.
     * Returns the number of matched spans.
     */
    private static int labelMatches(List<String> forms, List<OffsetToken> entityToks, int[] bio) {
        int matches = 0;
        int n = forms.size();
        int m = entityToks.size();
        int i = 0;
        while (i <= n - m) {
            boolean match = true;
            for (int j = 0; j < m; j++) {
                if (!forms.get(i + j).equalsIgnoreCase(entityToks.get(j).text())) {
                    match = false;
                    break;
                }
            }
            if (match) {
                bio[i] = LABEL_B;
                for (int j = 1; j < m; j++) bio[i + j] = LABEL_I;
                matches++;
                i += m;  // skip past this match
            } else {
                i++;
            }
        }
        return matches;
    }
}
