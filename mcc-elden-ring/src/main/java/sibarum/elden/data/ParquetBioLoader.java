package sibarum.elden.data;

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
 * Loads a Parquet file of pre-tokenized BIO-labeled sentences via DuckDB.
 *
 * Expects the Parquet schema to have two columns:
 *   - {@code tokens}: list&lt;string&gt; — surface forms of the sentence
 *   - {@code ner_tags}: list&lt;int&gt; — per-token labels, conventionally
 *     0=O, 1=B-ENT, 2=I-ENT
 *
 * Each row is one sentence. Mismatched array lengths are skipped with a
 * warning. Use {@link #discoverSchema} once to print the columns of an
 * unfamiliar file before adjusting column names.
 */
public final class ParquetBioLoader {

    public record Sentence(List<String> tokens, int[] bioLabels) {}

    private ParquetBioLoader() {}

    /** Print the column schema of a Parquet file. Diagnostic only. */
    public static void discoverSchema(Path file) throws SQLException, ClassNotFoundException {
        ensureDriver();
        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(
                     "DESCRIBE SELECT * FROM read_parquet('" + escape(file.toString()) + "')")) {
            System.out.println("Parquet schema for " + file + ":");
            while (rs.next()) {
                System.out.printf("  %-20s %s%n", rs.getString(1), rs.getString(2));
            }
        }
    }

    /**
     * Load all sentences (or up to {@code limit} if present). Column names
     * default to {@code tokens} and {@code ner_tags}; override if the file
     * uses different names.
     */
    public static List<Sentence> load(Path file, OptionalInt limit) throws SQLException, ClassNotFoundException {
        return load(file, "tokens", "ner_tags", limit);
    }

    public static List<Sentence> load(Path file, String tokensCol, String tagsCol, OptionalInt limit)
            throws SQLException, ClassNotFoundException {
        ensureDriver();
        List<Sentence> out = new ArrayList<>();
        String sql = "SELECT " + tokensCol + ", " + tagsCol
                + " FROM read_parquet('" + escape(file.toString()) + "')"
                + limit.stream().mapToObj(n -> " LIMIT " + n).findFirst().orElse("");

        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {

            int rowsSkipped = 0;
            while (rs.next()) {
                java.sql.Array tokensArr = rs.getArray(1);
                Object[] tokensRaw = (Object[]) tokensArr.getArray();
                java.sql.Array tagsArr = rs.getArray(2);
                Object[] tagsRaw = (Object[]) tagsArr.getArray();

                if (tokensRaw.length != tagsRaw.length) {
                    rowsSkipped++;
                    continue;
                }

                List<String> tokens = new ArrayList<>(tokensRaw.length);
                for (Object o : tokensRaw) tokens.add(o == null ? "" : o.toString());

                int[] tags = new int[tagsRaw.length];
                for (int i = 0; i < tags.length; i++) {
                    tags[i] = ((Number) tagsRaw[i]).intValue();
                }

                out.add(new Sentence(tokens, tags));
            }
            if (rowsSkipped > 0) {
                System.err.println("ParquetBioLoader: skipped " + rowsSkipped + " malformed rows");
            }
        }
        return out;
    }

    private static void ensureDriver() throws ClassNotFoundException {
        Class.forName("org.duckdb.DuckDBDriver");
    }

    private static String escape(String s) {
        return s.replace("'", "''");
    }
}
