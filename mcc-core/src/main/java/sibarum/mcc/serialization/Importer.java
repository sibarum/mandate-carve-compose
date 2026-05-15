package sibarum.mcc.serialization;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationEdge;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.graph.substrate.TransformationNode;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.primitive.PrimitiveRegistry;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Inverse of {@link Exporter}. Reads {@code component.mcc/graph.json}
 * and {@code component.mcc/params.bin}, verifies the SHA-256 in the
 * manifest matches the blob body, and reconstructs a runnable
 * {@link ComputationGraph} with all parameters restored.
 */
public final class Importer {

    private final PrimitiveRegistry registry;

    public Importer(PrimitiveRegistry registry) {
        this.registry = registry;
    }

    public record LoadedGraph(
            ComputationGraph graph,
            Map<String, RootBinding> rootInputs) {}

    public record RootBinding(CompGraphNode node, int slot, String name, String type) {}

    public LoadedGraph load(Path dir) throws IOException {
        Path graphPath = dir.resolve("graph.json");
        Path paramsPath = dir.resolve("params.bin");

        String graphJson = Files.readString(graphPath, StandardCharsets.UTF_8);
        GraphSchema schema = GraphSchemaCodec.fromJson(graphJson);
        if (schema.schemaVersion() != GraphSchema.CURRENT_VERSION) {
            throw new IOException("unsupported schema version: " + schema.schemaVersion()
                    + " (expected " + GraphSchema.CURRENT_VERSION + ")");
        }

        ParameterBlob.ReadResult blob;
        try (InputStream in = Files.newInputStream(paramsPath)) {
            blob = ParameterBlob.read(in);
        }
        String expectedSha = schema.parameters() == null ? null : schema.parameters().sha256();
        if (expectedSha != null && !expectedSha.equals(blob.sha256())) {
            throw new IOException("params.bin SHA-256 mismatch: expected "
                    + expectedSha + " got " + blob.sha256());
        }

        // Build substrate + Primitive instances.
        TransformationGraphBuilder builder = new TransformationGraphBuilder();
        Map<String, Primitive> primitivesById = new HashMap<>();
        for (GraphSchema.NodeDesc nd : schema.nodes()) {
            Primitive p = registry.create(nd.primitive(), nd.config());
            builder.addNode(nd.id(), p);
            primitivesById.put(nd.id(), p);
        }
        TransformationGraph tg = builder.build();

        // Build CompGraphNodes.
        Map<String, CompGraphNode> compNodes = new HashMap<>();
        for (GraphSchema.NodeDesc nd : schema.nodes()) {
            TransformationNode tn = tg.node(nd.id());
            compNodes.put(nd.id(), new CompGraphNode(nd.id(), tn));
        }

        // Wire edges.
        for (GraphSchema.EdgeDesc ed : schema.edges()) {
            CompGraphNode from = compNodes.get(ed.from());
            CompGraphNode to = compNodes.get(ed.to());
            TransformationEdge tEdge = tg.edge(from.tNode(), to.tNode());
            if (tEdge == null) {
                throw new IOException("import: type-incompatible edge "
                        + ed.from() + " -> " + ed.to());
            }
            to.wire(ed.slot(), new SlotSource(from, tEdge));
        }

        CompGraphNode terminal = compNodes.get(schema.terminal());
        if (terminal == null) {
            throw new IOException("import: terminal node '" + schema.terminal() + "' not found");
        }
        List<CompGraphNode> orderedNodes = new ArrayList<>();
        for (GraphSchema.NodeDesc nd : schema.nodes()) orderedNodes.add(compNodes.get(nd.id()));
        ComputationGraph cg = new ComputationGraph(orderedNodes, terminal);

        // Restore parameters per node.
        Map<String, Map<String, Parameterized.NamedTensor>> tensorsByNode = new HashMap<>();
        if (schema.parameters() != null) {
            for (GraphSchema.TensorEntry te : schema.parameters().tensors()) {
                double[] data = ParameterBlob.decode(blob.body(), te.offset(), te.length());
                tensorsByNode
                        .computeIfAbsent(te.node(), k -> new HashMap<>())
                        .put(te.name(), new Parameterized.NamedTensor(te.name(), te.shape(), data));
            }
        }
        for (Map.Entry<String, Map<String, Parameterized.NamedTensor>> entry : tensorsByNode.entrySet()) {
            Primitive p = primitivesById.get(entry.getKey());
            if (p instanceof Parameterized pp) {
                pp.loadParameters(entry.getValue());
            }
        }

        // Build the root-input map.
        Map<String, RootBinding> rootMap = new HashMap<>();
        if (schema.rootInputs() != null) {
            for (GraphSchema.RootInputDesc rd : schema.rootInputs()) {
                CompGraphNode node = compNodes.get(rd.node());
                if (node == null) {
                    throw new IOException("import: root-input node '" + rd.node() + "' not found");
                }
                rootMap.put(rd.name(), new RootBinding(node, rd.slot(), rd.name(), rd.type()));
            }
        }

        return new LoadedGraph(cg, rootMap);
    }
}
