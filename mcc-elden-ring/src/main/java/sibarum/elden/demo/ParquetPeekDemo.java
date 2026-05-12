package sibarum.elden.demo;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;

/**
 * Tiny diagnostic: peek at the schema and first few rows of a Parquet file.
 * Reusable for inspecting any new dataset before deciding how to ingest it.
 *
 * Usage:
 *   ParquetPeekDemo &lt;path-to-parquet&gt; [num-rows]
 */
public final class ParquetPeekDemo {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: ParquetPeekDemo <path-to-parquet> [num-rows]");
            System.exit(1);
        }
        Path file = Path.of(args[0]);
        int n = args.length >= 2 ? Integer.parseInt(args[1]) : 5;

        Class.forName("org.duckdb.DuckDBDriver");
        String src = "read_parquet('" + file.toString().replace("'", "''") + "')";
        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement()) {

            System.out.println("=== schema ===");
            try (ResultSet rs = stmt.executeQuery("DESCRIBE SELECT * FROM " + src)) {
                while (rs.next()) {
                    System.out.printf("  %-25s %s%n", rs.getString(1), rs.getString(2));
                }
            }
            System.out.println();

            System.out.println("=== row count ===");
            try (ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM " + src)) {
                if (rs.next()) System.out.println("  " + rs.getLong(1));
            }
            System.out.println();

            System.out.println("=== first " + n + " rows ===");
            try (ResultSet rs = stmt.executeQuery("SELECT * FROM " + src + " LIMIT " + n)) {
                ResultSetMetaData md = rs.getMetaData();
                int cols = md.getColumnCount();
                int row = 1;
                while (rs.next()) {
                    System.out.println("--- row " + row++ + " ---");
                    for (int c = 1; c <= cols; c++) {
                        Object v = rs.getObject(c);
                        String s;
                        if (v instanceof java.sql.Array a) {
                            Object[] arr = (Object[]) a.getArray();
                            StringBuilder sb = new StringBuilder("[");
                            for (int i = 0; i < arr.length; i++) {
                                if (i > 0) sb.append(", ");
                                sb.append(arr[i]);
                            }
                            sb.append("]");
                            s = sb.toString();
                        } else {
                            s = String.valueOf(v);
                        }
                        if (s.length() > 240) s = s.substring(0, 240) + "...";
                        System.out.printf("  %-25s %s%n", md.getColumnName(c), s);
                    }
                }
            }
        }
    }
}
