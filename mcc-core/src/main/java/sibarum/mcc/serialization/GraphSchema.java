package sibarum.mcc.serialization;

import java.util.List;
import java.util.Map;

/**
 * Reflection-free POJO of a serialized {@code component.mcc} graph.
 * Marshalling lives in {@link GraphSchemaCodec}; this file is just
 * the data shape — no annotations, no Jackson dependency.
 *
 * <pre>
 * {
 *   "schemaVersion": 1,
 *   "nodes": [{ "id", "primitive", "config" }],
 *   "edges": [{ "from", "to", "slot" }],
 *   "rootInputs": [{ "node", "slot", "name", "type" }],
 *   "terminal": "node-id",
 *   "parameters": {
 *     "sha256": "...",
 *     "tensors": [{ "node": "...", "name": "...", "shape": [...], "offset": int, "length": int }]
 *   }
 * }
 * </pre>
 *
 * <p>Field names in the JSON are the record component names; tooling
 * that produces this format must use the same names.
 */
public record GraphSchema(
        int schemaVersion,
        List<NodeDesc> nodes,
        List<EdgeDesc> edges,
        List<RootInputDesc> rootInputs,
        String terminal,
        ParameterManifest parameters) {

    public static final int CURRENT_VERSION = 1;

    public record NodeDesc(String id, String primitive, Map<String, Object> config) {}

    public record EdgeDesc(String from, String to, int slot) {}

    public record RootInputDesc(String node, int slot, String name, String type) {}

    public record ParameterManifest(String sha256, List<TensorEntry> tensors) {}

    public record TensorEntry(String node, String name, int[] shape, long offset, long length) {}
}
