package sibarum.mcc.serialization;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Primitive;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writes a {@link ComputationGraph} + its trained parameters to a
 * {@code component.mcc} directory:
 *
 * <pre>
 *   component.mcc/
 *   ├── graph.json     (topology + tensor manifest)
 *   └── params.bin     (framed binary tensors, body-only SHA-256 in manifest)
 * </pre>
 *
 * <p>Per-node parameters are collected by walking the graph for
 * {@link Parameterized} primitives. Each unique trainable identity is
 * written once; if the same Trainable backs multiple nodes (weight
 * sharing), only one snapshot appears in the blob and the manifest
 * cross-references it.
 */
public final class Exporter {

    public Exporter() {
    }

    /** Description of a named root input the runtime should accept. */
    public record RootInput(String name, CompGraphNode node, int slot) {}

    public void export(ComputationGraph graph, List<RootInput> rootInputs, Path outDir) throws IOException {
        Files.createDirectories(outDir);

        List<GraphSchema.NodeDesc> nodes = new ArrayList<>();
        List<GraphSchema.EdgeDesc> edges = new ArrayList<>();
        List<GraphSchema.RootInputDesc> roots = new ArrayList<>();

        for (CompGraphNode n : graph.nodes()) {
            Primitive p = n.tNode().primitive();
            Map<String, Object> cfg = (p instanceof Configurable c) ? c.config() : Map.of();
            nodes.add(new GraphSchema.NodeDesc(n.id(), p.name(), cfg));

            for (int slot = 0; slot < n.slotCount(); slot++) {
                SlotSource src = n.slot(slot);
                if (src != null) {
                    edges.add(new GraphSchema.EdgeDesc(src.source().id(), n.id(), slot));
                }
            }
        }

        for (RootInput ri : rootInputs) {
            String type = ri.node().tNode().inputTypes().get(ri.slot()).name();
            roots.add(new GraphSchema.RootInputDesc(ri.node().id(), ri.slot(), ri.name(), type));
        }

        // Collect unique parameterized primitives (dedup by identity).
        IdentityHashMap<Object, ParameterizedNode> uniqueParams = new IdentityHashMap<>();
        for (CompGraphNode n : graph.nodes()) {
            Primitive p = n.tNode().primitive();
            if (p instanceof Parameterized pp) {
                Object id = p;
                uniqueParams.putIfAbsent(id, new ParameterizedNode(n.id(), pp));
            }
        }

        // Snapshot tensors and write params.bin.
        List<ParameterBlob.NamedNodeTensor> tensors = new ArrayList<>();
        for (ParameterizedNode pn : uniqueParams.values()) {
            for (Parameterized.NamedTensor t : pn.p().parameters()) {
                tensors.add(new ParameterBlob.NamedNodeTensor(
                        pn.nodeId(), t.name(), t.shape(), t.data()));
            }
        }

        Path paramsPath = outDir.resolve("params.bin");
        ParameterBlob.WriteResult wr;
        try (OutputStream out = Files.newOutputStream(paramsPath)) {
            wr = ParameterBlob.write(tensors, out);
        }

        GraphSchema schema = new GraphSchema(
                GraphSchema.CURRENT_VERSION,
                nodes,
                edges,
                roots,
                graph.terminal().id(),
                new GraphSchema.ParameterManifest(wr.sha256(), wr.entries())
        );

        Path graphPath = outDir.resolve("graph.json");
        Files.writeString(graphPath, GraphSchemaCodec.toJson(schema), StandardCharsets.UTF_8);
    }

    private record ParameterizedNode(String nodeId, Parameterized p) {}
}
