package sibarum.elden.data;

import sibarum.elden.data.ParquetBioLoader.Sentence;

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
 * Loads BIO-labeled training sentences from {@code NLP-KnowledgeGraph.csv},
 * a generic-English relation-extraction corpus with per-token SRC/REL/TGT/O
 * annotations. Each row's {@code tokens} and {@code tags} are Python-list
 * literal strings; this loader parses them and maps:
 *
 * <ul>
 *   <li>contiguous SRC tokens → one B-normal + I-normal* span</li>
 *   <li>contiguous TGT tokens → one B-normal + I-normal* span</li>
 *   <li>REL tokens             → O (these are predicate verbs, not entities)</li>
 *   <li>O tokens               → O</li>
 * </ul>
 *
 * This is iter 18's "normal" / generic-English contribution to the
 * three-tier corpus gradient. The signal is noisy (the source dataset is
 * not perfectly annotated) but provides large-volume training that a
 * token is part of *some* nominal-role thing in standard English prose.
 */
public final class KnowledgeGraphBioLoader {

    private static final int B = EntityClasses.B_NORMAL;
    private static final int I = EntityClasses.I_NORMAL;
    private static final int O = EntityClasses.O;

    private KnowledgeGraphBioLoader() {}

    public static List<Sentence> load(Path file, OptionalInt limit, long seed)
            throws SQLException, ClassNotFoundException {
        Class.forName("org.duckdb.DuckDBDriver");
        String src = "read_csv_auto('" + file.toString().replace("'", "''") + "')";
        String sql;
        if (limit.isPresent()) {
            sql = "SELECT tokens, tags FROM " + src
                    + " WHERE tokens IS NOT NULL AND tags IS NOT NULL"
                    + " USING SAMPLE " + limit.getAsInt() + " ROWS (reservoir, " + seed + ")";
        } else {
            sql = "SELECT tokens, tags FROM " + src
                    + " WHERE tokens IS NOT NULL AND tags IS NOT NULL";
        }

        List<Sentence> out = new ArrayList<>();
        int totalRows = 0, skipped = 0;
        long totalTokens = 0, entityTokens = 0;
        int totalSpans = 0;

        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                totalRows++;
                List<String> tokens = parsePyList(rs.getString(1));
                List<String> tags = parsePyList(rs.getString(2));
                if (tokens.isEmpty() || tokens.size() != tags.size()) {
                    skipped++; continue;
                }

                int[] bio = new int[tokens.size()];
                int spans = labelSrcTgt(tags, bio);
                if (spans == 0) { skipped++; continue; }
                out.add(new Sentence(tokens, bio));
                totalSpans += spans;
                totalTokens += tokens.size();
                for (int b : bio) if (b != O) entityTokens++;
            }
        }

        System.out.printf("KnowledgeGraphBioLoader: %,d rows -> %,d labeled (%,d skipped)%n",
                totalRows, out.size(), skipped);
        System.out.printf("  %,d tokens, %,d entity (%.1f%%), %,d spans%n",
                totalTokens, entityTokens,
                100.0 * entityTokens / Math.max(totalTokens, 1),
                totalSpans);
        return out;
    }

    /**
     * Convert a per-token tag sequence {O, SRC, REL, TGT} into a B-normal/
     * I-normal/O BIO sequence. SRC and TGT runs each become their own spans;
     * REL is treated as O (it's a relation predicate, not an entity).
     * Returns the number of spans produced.
     */
    private static int labelSrcTgt(List<String> tags, int[] bio) {
        int spans = 0;
        String prev = "O";
        for (int i = 0; i < tags.size(); i++) {
            String t = tags.get(i);
            boolean entityClass = "SRC".equals(t) || "TGT".equals(t);
            if (entityClass) {
                if (!t.equals(prev)) {
                    bio[i] = B;
                    spans++;
                } else {
                    bio[i] = I;
                }
            } else {
                bio[i] = O;
            }
            prev = t;
        }
        return spans;
    }

    /**
     * Parse a Python-list literal of strings into a Java list. Handles both
     * single and double quotes and basic backslash escapes. Returns an empty
     * list on malformed input rather than throwing.
     */
    static List<String> parsePyList(String s) {
        List<String> out = new ArrayList<>();
        if (s == null) return out;
        s = s.trim();
        if (s.startsWith("[")) s = s.substring(1);
        if (s.endsWith("]")) s = s.substring(0, s.length() - 1);
        int i = 0, n = s.length();
        while (i < n) {
            while (i < n && (s.charAt(i) == ',' || s.charAt(i) == ' ' || s.charAt(i) == '\t')) i++;
            if (i >= n) break;
            char quote = s.charAt(i);
            if (quote != '\'' && quote != '"') { out.clear(); return out; }
            i++;
            StringBuilder sb = new StringBuilder();
            while (i < n) {
                char c = s.charAt(i);
                if (c == '\\' && i + 1 < n) {
                    sb.append(s.charAt(i + 1));
                    i += 2;
                } else if (c == quote) {
                    break;
                } else {
                    sb.append(c);
                    i++;
                }
            }
            if (i >= n) { out.clear(); return out; }
            i++;
            out.add(sb.toString());
        }
        return out;
    }
}
