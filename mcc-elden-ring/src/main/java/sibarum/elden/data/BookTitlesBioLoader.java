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
import java.util.Random;

/**
 * Loads book titles from a Parquet file (schema: {@code name, persons, year,
 * field_name}) and emits BIO-labeled training {@link Sentence} records.
 *
 * Each title is wrapped in one of a small set of simple sentence templates
 * chosen at random — sometimes bare ("Rune of Death ."), sometimes wrapped
 * ("She wrote Rune of Death .") — and the title's tokens are labeled B then
 * I+ while every other token (template chrome and final punctuation) is O.
 *
 * The point is span coherence on multi-word capitalized phrases with
 * INTERNAL function words ("of the", "for the", "to the") — exactly the
 * structural pattern the entity tagger has been splitting on (e.g.
 * "Remembrance of the Baleful Shadow" → "Remembrance of" + "Baleful
 * Shadow"). 1.48M book titles is a high-volume corpus of that pattern.
 */
public final class BookTitlesBioLoader {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;

    /** Paired before/after wrappers; index i in BEFORE matches index i in AFTER. */
    private static final String[] BEFORE = {
            "",                       // bare title only
            "",                       // bare title only (extra weight on the no-wrap case)
            "She wrote ",
            "He read ",
            "The book ",
            "I recommend ",
            "According to ",
            "Many cite ",
            "They published ",
            "We studied ",
    };
    private static final String[] AFTER = {
            "",
            " .",
            " .",
            " last year .",
            " was popular .",
            " highly .",
            " , this is true .",
            " as a standard .",
            " in 1990 .",
            " for the course .",
    };

    private BookTitlesBioLoader() {}

    /**
     * @param file       Parquet file path
     * @param limit      max titles to load (USING SAMPLE for random subset);
     *                   empty means all rows
     * @param seed       RNG seed for template selection
     */
    public static List<Sentence> load(Path file, OptionalInt limit, long seed)
            throws SQLException, ClassNotFoundException {
        Class.forName("org.duckdb.DuckDBDriver");
        String src = "read_parquet('" + file.toString().replace("'", "''") + "')";
        String sql;
        if (limit.isPresent()) {
            sql = "SELECT name FROM " + src + " WHERE name IS NOT NULL"
                    + " USING SAMPLE " + limit.getAsInt() + " ROWS (reservoir, " + seed + ")";
        } else {
            sql = "SELECT name FROM " + src + " WHERE name IS NOT NULL";
        }

        Random rng = new Random(seed);
        List<Sentence> out = new ArrayList<>();
        int skipped = 0;
        long titleTokens = 0;
        long totalTokens = 0;
        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                String name = rs.getString(1);
                if (name == null || name.isBlank()) { skipped++; continue; }
                name = name.trim();
                int idx = rng.nextInt(BEFORE.length);
                String before = BEFORE[idx];
                String after = AFTER[idx];
                String full = before + name + after;
                List<OffsetToken> toks = OffsetTokenizer.tokenize(full);
                if (toks.isEmpty()) { skipped++; continue; }
                int titleStartChar = before.length();
                int titleEndChar = titleStartChar + name.length();
                List<String> forms = new ArrayList<>(toks.size());
                int[] bio = new int[toks.size()];
                int firstTitleTokIdx = -1;
                for (int i = 0; i < toks.size(); i++) {
                    OffsetToken t = toks.get(i);
                    forms.add(t.text());
                    if (t.startChar() >= titleStartChar && t.endChar() <= titleEndChar) {
                        if (firstTitleTokIdx < 0) firstTitleTokIdx = i;
                        bio[i] = (i == firstTitleTokIdx) ? LABEL_B : LABEL_I;
                        titleTokens++;
                    } else {
                        bio[i] = LABEL_O;
                    }
                    totalTokens++;
                }
                if (firstTitleTokIdx < 0) { skipped++; continue; }
                out.add(new Sentence(forms, bio));
            }
        }
        if (skipped > 0) {
            System.err.println("BookTitlesBioLoader: skipped " + skipped + " rows (null/blank/no-title-tokens)");
        }
        System.out.printf("BookTitlesBioLoader: loaded %,d sentences (%,d tokens, %,d title tokens = %.1f%%)%n",
                out.size(), totalTokens, titleTokens, 100.0 * titleTokens / Math.max(totalTokens, 1));
        return out;
    }
}
