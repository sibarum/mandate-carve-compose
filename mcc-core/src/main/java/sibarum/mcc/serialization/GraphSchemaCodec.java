package sibarum.mcc.serialization;

import sibarum.mcc.serialization.GraphSchema.EdgeDesc;
import sibarum.mcc.serialization.GraphSchema.NodeDesc;
import sibarum.mcc.serialization.GraphSchema.ParameterManifest;
import sibarum.mcc.serialization.GraphSchema.RootInputDesc;
import sibarum.mcc.serialization.GraphSchema.TensorEntry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reflection-free codec for {@link GraphSchema}. Pairs with
 * {@link JsonWriter} / {@link JsonReader} to produce / consume the
 * {@code graph.json} document in a Graal native-image friendly way.
 *
 * <p>Schema field names are stable wire identifiers. Any rename here
 * is a {@code schemaVersion} bump.
 */
public final class GraphSchemaCodec {

    private GraphSchemaCodec() {}

    public static String toJson(GraphSchema schema) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("schemaVersion", schema.schemaVersion());
        root.put("nodes", encodeNodes(schema.nodes()));
        root.put("edges", encodeEdges(schema.edges()));
        root.put("rootInputs", encodeRootInputs(schema.rootInputs()));
        root.put("terminal", schema.terminal());
        root.put("parameters", encodeParameters(schema.parameters()));
        JsonWriter w = new JsonWriter(true);
        w.writeValue(root);
        return w.result();
    }

    @SuppressWarnings("unchecked")
    public static GraphSchema fromJson(String json) {
        Object parsed = JsonReader.parse(json);
        if (!(parsed instanceof Map<?, ?> raw)) {
            throw new IllegalArgumentException("graph.json root must be an object");
        }
        Map<String, Object> m = (Map<String, Object>) raw;
        int version = ((Number) require(m, "schemaVersion")).intValue();
        List<NodeDesc> nodes = decodeNodes(asList(m.get("nodes")));
        List<EdgeDesc> edges = decodeEdges(asList(m.get("edges")));
        List<RootInputDesc> rootInputs = decodeRootInputs(asList(m.get("rootInputs")));
        String terminal = (String) require(m, "terminal");
        ParameterManifest params = decodeParameters(m.get("parameters"));
        return new GraphSchema(version, nodes, edges, rootInputs, terminal, params);
    }

    // ---- encoders ----

    private static List<Object> encodeNodes(List<NodeDesc> nodes) {
        List<Object> out = new ArrayList<>(nodes.size());
        for (NodeDesc n : nodes) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", n.id());
            m.put("primitive", n.primitive());
            m.put("config", n.config());
            out.add(m);
        }
        return out;
    }

    private static List<Object> encodeEdges(List<EdgeDesc> edges) {
        List<Object> out = new ArrayList<>(edges.size());
        for (EdgeDesc e : edges) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("from", e.from());
            m.put("to", e.to());
            m.put("slot", e.slot());
            out.add(m);
        }
        return out;
    }

    private static List<Object> encodeRootInputs(List<RootInputDesc> roots) {
        List<Object> out = new ArrayList<>(roots.size());
        for (RootInputDesc r : roots) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("node", r.node());
            m.put("slot", r.slot());
            m.put("name", r.name());
            m.put("type", r.type());
            out.add(m);
        }
        return out;
    }

    private static Map<String, Object> encodeParameters(ParameterManifest pm) {
        if (pm == null) return null;
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("sha256", pm.sha256());
        List<Object> tensors = new ArrayList<>(pm.tensors().size());
        for (TensorEntry t : pm.tensors()) {
            Map<String, Object> tm = new LinkedHashMap<>();
            tm.put("node", t.node());
            tm.put("name", t.name());
            tm.put("shape", t.shape());
            tm.put("offset", t.offset());
            tm.put("length", t.length());
            tensors.add(tm);
        }
        m.put("tensors", tensors);
        return m;
    }

    // ---- decoders ----

    @SuppressWarnings("unchecked")
    private static List<NodeDesc> decodeNodes(List<Object> raw) {
        List<NodeDesc> out = new ArrayList<>(raw.size());
        for (Object o : raw) {
            Map<String, Object> m = (Map<String, Object>) o;
            Map<String, Object> cfg = m.get("config") == null
                    ? Map.of()
                    : (Map<String, Object>) m.get("config");
            out.add(new NodeDesc((String) require(m, "id"),
                    (String) require(m, "primitive"),
                    cfg));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static List<EdgeDesc> decodeEdges(List<Object> raw) {
        List<EdgeDesc> out = new ArrayList<>(raw.size());
        for (Object o : raw) {
            Map<String, Object> m = (Map<String, Object>) o;
            out.add(new EdgeDesc((String) require(m, "from"),
                    (String) require(m, "to"),
                    ((Number) require(m, "slot")).intValue()));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static List<RootInputDesc> decodeRootInputs(List<Object> raw) {
        List<RootInputDesc> out = new ArrayList<>(raw.size());
        for (Object o : raw) {
            Map<String, Object> m = (Map<String, Object>) o;
            out.add(new RootInputDesc((String) require(m, "node"),
                    ((Number) require(m, "slot")).intValue(),
                    (String) require(m, "name"),
                    (String) require(m, "type")));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static ParameterManifest decodeParameters(Object raw) {
        if (raw == null) return null;
        Map<String, Object> m = (Map<String, Object>) raw;
        String sha = (String) m.get("sha256");
        List<Object> tensors = asList(m.get("tensors"));
        List<TensorEntry> entries = new ArrayList<>(tensors.size());
        for (Object o : tensors) {
            Map<String, Object> tm = (Map<String, Object>) o;
            int[] shape = decodeIntArray(asList(tm.get("shape")));
            entries.add(new TensorEntry(
                    (String) require(tm, "node"),
                    (String) require(tm, "name"),
                    shape,
                    ((Number) require(tm, "offset")).longValue(),
                    ((Number) require(tm, "length")).longValue()));
        }
        return new ParameterManifest(sha, entries);
    }

    private static int[] decodeIntArray(List<Object> raw) {
        int[] out = new int[raw.size()];
        for (int i = 0; i < raw.size(); i++) out[i] = ((Number) raw.get(i)).intValue();
        return out;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object o) {
        if (o == null) return List.of();
        return (List<Object>) o;
    }

    private static Object require(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (v == null) {
            throw new IllegalArgumentException("graph.json: missing required field '" + key + "'");
        }
        return v;
    }
}
