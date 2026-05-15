package sibarum.mcc.serialization;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sanity tests for the reflection-free JSON utilities. Locks the
 * primitive types they support and confirms write→read round-trips.
 */
class JsonRoundTripTest {

    @Test
    void primitivesRoundTrip() {
        assertEquals(Boolean.TRUE, JsonReader.parse("true"));
        assertEquals(Boolean.FALSE, JsonReader.parse("false"));
        assertNull(JsonReader.parse("null"));
        assertEquals(42L, JsonReader.parse("42"));
        assertEquals(-7L, JsonReader.parse("-7"));
        assertEquals(3.14, (Double) JsonReader.parse("3.14"), 1e-12);
        assertEquals(1e10, (Double) JsonReader.parse("1e10"), 1e-3);
        assertEquals("hello", JsonReader.parse("\"hello\""));
        assertEquals("a\"b\\c\n", JsonReader.parse("\"a\\\"b\\\\c\\n\""));
    }

    @Test
    void writerEmitsValidJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", "x");
        m.put("count", 3);
        m.put("ratio", 0.5);
        m.put("flag", true);
        m.put("optional", null);
        m.put("tags", List.of("a", "b"));
        m.put("shape", new int[] { 2, 3 });

        JsonWriter w = new JsonWriter(false);
        w.writeValue(m);
        String out = w.result();

        Object parsed = JsonReader.parse(out);
        @SuppressWarnings("unchecked")
        Map<String, Object> back = (Map<String, Object>) parsed;
        assertEquals("x", back.get("name"));
        assertEquals(3L, back.get("count"));
        assertEquals(0.5, (Double) back.get("ratio"), 1e-12);
        assertEquals(Boolean.TRUE, back.get("flag"));
        assertNull(back.get("optional"));
        assertEquals(List.of("a", "b"), back.get("tags"));
        // int[] round-trips as List<Long>.
        List<?> shape = (List<?>) back.get("shape");
        assertEquals(2, shape.size());
        assertEquals(2L, shape.get(0));
        assertEquals(3L, shape.get(1));
    }

    @Test
    void preservesInsertionOrder() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("zebra", 1);
        m.put("apple", 2);
        m.put("middle", 3);

        JsonWriter w = new JsonWriter(false);
        w.writeValue(m);
        @SuppressWarnings("unchecked")
        Map<String, Object> back = (Map<String, Object>) JsonReader.parse(w.result());
        assertArrayEquals(new String[] { "zebra", "apple", "middle" },
                back.keySet().toArray(new String[0]));
    }

    @Test
    void rejectsNaNAndInfinity() {
        JsonWriter w = new JsonWriter(false);
        assertThrows(IllegalArgumentException.class, () -> w.writeValue(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> w.writeValue(Double.POSITIVE_INFINITY));
    }

    @Test
    void rejectsTrailingContent() {
        assertThrows(IllegalArgumentException.class, () -> JsonReader.parse("123 garbage"));
        // But whitespace around root is fine.
        assertEquals(123L, JsonReader.parse("  123  "));
    }

    @Test
    void rejectsUnterminatedString() {
        assertThrows(IllegalArgumentException.class, () -> JsonReader.parse("\"hello"));
    }

    @Test
    void graphSchemaCodecRoundTrip() {
        GraphSchema original = new GraphSchema(
                1,
                List.of(
                        new GraphSchema.NodeDesc("c-mlp", "mlp-block",
                                Map.of("sizes", List.of(2, 4, 1))),
                        new GraphSchema.NodeDesc("c-add", "add", Map.of())
                ),
                List.of(
                        new GraphSchema.EdgeDesc("c-mlp", "c-add", 0)
                ),
                List.of(
                        new GraphSchema.RootInputDesc("c-mlp", 0, "x", "MATRIX")
                ),
                "c-add",
                new GraphSchema.ParameterManifest(
                        "abc123",
                        List.of(
                                new GraphSchema.TensorEntry("c-mlp", "W0",
                                        new int[] { 4, 2 }, 0L, 64L),
                                new GraphSchema.TensorEntry("c-mlp", "b0",
                                        new int[] { 4 }, 64L, 32L)
                        ))
        );

        String json = GraphSchemaCodec.toJson(original);
        GraphSchema back = GraphSchemaCodec.fromJson(json);

        assertEquals(original.schemaVersion(), back.schemaVersion());
        assertEquals(original.terminal(), back.terminal());
        assertEquals(original.nodes().size(), back.nodes().size());
        assertEquals(original.edges().size(), back.edges().size());
        assertEquals(original.rootInputs().size(), back.rootInputs().size());
        assertEquals(original.parameters().sha256(), back.parameters().sha256());
        assertEquals(original.parameters().tensors().size(),
                back.parameters().tensors().size());

        GraphSchema.NodeDesc n0 = back.nodes().get(0);
        assertEquals("c-mlp", n0.id());
        assertEquals("mlp-block", n0.primitive());
        @SuppressWarnings("unchecked")
        List<Number> sizes = (List<Number>) n0.config().get("sizes");
        assertEquals(3, sizes.size());

        GraphSchema.TensorEntry t0 = back.parameters().tensors().get(0);
        assertArrayEquals(new int[] { 4, 2 }, t0.shape());
        assertEquals(0L, t0.offset());
        assertEquals(64L, t0.length());
    }
}
