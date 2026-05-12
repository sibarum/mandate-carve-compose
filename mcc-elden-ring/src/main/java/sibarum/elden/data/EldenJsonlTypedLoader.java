package sibarum.elden.data;

import sibarum.elden.annotation.EntityType;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads typed-span training examples from {@code elden_ring_final_train.jsonl}.
 *
 * For each row:
 * <ol>
 *   <li>Map {@code metadata.entity_type} to our {@link EntityType} taxonomy
 *       via {@link #JSONL_TYPE_MAP}. Unmapped rows are skipped.</li>
 *   <li>Tokenize {@code output} text with {@link OffsetTokenizer}.</li>
 *   <li>Find every token-aligned, case-insensitive match of
 *       {@code metadata.entity_name} in the tokenized output.</li>
 *   <li>For each match, emit a {@link TypedSpanExample}.</li>
 * </ol>
 *
 * The JSONL has 10 entity_type values; all 10 map to one of {ARTIFACT,
 * CHARACTER, PLACE, CONCEPT}. EVENT / FACTION / ERA are not represented in
 * the JSONL — those come from hand-annotated supervision only.
 */
public final class EldenJsonlTypedLoader {

    /** Maps JSONL {@code entity_type} strings to our {@link EntityType} values. */
    public static final Map<String, EntityType> JSONL_TYPE_MAP;
    static {
        Map<String, EntityType> m = new LinkedHashMap<>();
        m.put("weapon", EntityType.ARTIFACT);
        m.put("armor", EntityType.ARTIFACT);
        m.put("boss", EntityType.CHARACTER);
        m.put("npc", EntityType.CHARACTER);
        m.put("creature", EntityType.CHARACTER);
        m.put("location", EntityType.PLACE);
        m.put("incantation", EntityType.CONCEPT);
        m.put("sorcery", EntityType.CONCEPT);
        m.put("ash_of_war", EntityType.CONCEPT);
        m.put("skill", EntityType.CONCEPT);
        JSONL_TYPE_MAP = Map.copyOf(m);
    }

    private EldenJsonlTypedLoader() {}

    public static List<TypedSpanExample> load(Path file) throws SQLException, ClassNotFoundException {
        Class.forName("org.duckdb.DuckDBDriver");
        String src = "read_json_auto('" + file.toString().replace("'", "''") + "', format='nd')";
        String sql = "SELECT output, metadata.entity_name AS entity_name,"
                + " metadata.entity_type AS entity_type FROM " + src
                + " WHERE output IS NOT NULL AND metadata.entity_name IS NOT NULL"
                + " AND metadata.entity_type IS NOT NULL";

        List<TypedSpanExample> out = new ArrayList<>();
        Map<EntityType, Integer> perTypeCounts = new LinkedHashMap<>();
        int totalRows = 0, matchedRows = 0, skippedNoType = 0, skippedNoMatch = 0;

        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                totalRows++;
                String output = rs.getString(1);
                String entity = rs.getString(2);
                String typeStr = rs.getString(3);
                if (output == null || entity == null || typeStr == null) { skippedNoType++; continue; }
                EntityType type = JSONL_TYPE_MAP.get(typeStr.trim());
                if (type == null) { skippedNoType++; continue; }

                List<OffsetToken> outputToks = OffsetTokenizer.tokenize(output);
                List<OffsetToken> entityToks = OffsetTokenizer.tokenize(entity.trim());
                if (outputToks.isEmpty() || entityToks.isEmpty()) { skippedNoMatch++; continue; }

                List<String> forms = new ArrayList<>(outputToks.size());
                for (OffsetToken t : outputToks) forms.add(t.text());

                int matches = 0;
                int n = forms.size();
                int m = entityToks.size();
                int i = 0;
                while (i <= n - m) {
                    boolean match = true;
                    for (int j = 0; j < m; j++) {
                        if (!forms.get(i + j).equalsIgnoreCase(entityToks.get(j).text())) {
                            match = false; break;
                        }
                    }
                    if (match) {
                        out.add(new TypedSpanExample(forms, i, i + m - 1, type));
                        matches++;
                        i += m;
                    } else {
                        i++;
                    }
                }
                if (matches == 0) { skippedNoMatch++; continue; }
                matchedRows++;
                perTypeCounts.merge(type, matches, Integer::sum);
            }
        }

        System.out.printf("EldenJsonlTypedLoader: %,d rows -> %,d matched (%,d typed spans)%n",
                totalRows, matchedRows, out.size());
        System.out.printf("  skipped: %,d (no/unmapped type), %,d (entity not found in output)%n",
                skippedNoType, skippedNoMatch);
        for (var e : perTypeCounts.entrySet()) {
            System.out.printf("    %-12s : %,d%n", e.getKey(), e.getValue());
        }
        return out;
    }
}
