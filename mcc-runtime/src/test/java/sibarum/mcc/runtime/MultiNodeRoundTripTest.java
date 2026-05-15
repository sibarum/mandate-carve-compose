package sibarum.mcc.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import sibarum.mcc.embedding.Embed;
import sibarum.mcc.embedding.SymbolEmbeddingTable;
import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.Softmax;
import sibarum.mcc.serialization.Exporter;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;
import sibarum.mcc.value.Value;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Hardens the export format on a graph more complex than the MVP's
 * single-MlpBlock case. Tests:
 *
 * <ul>
 *   <li>Embed (Trainable + Parameterized + Configurable with dynamic
 *       symbols list) round-trips.</li>
 *   <li>Edge serialization on a multi-node chain (3 nodes, 2 edges)
 *       reconstructs correctly.</li>
 *   <li>Stateless ops between trainables (Softmax) don't lose their
 *       wiring.</li>
 *   <li>Multiple parameterized nodes (Embed + Linear) each get their
 *       own tensor block in {@code params.bin}.</li>
 * </ul>
 */
class MultiNodeRoundTripTest {

    @TempDir
    Path tempDir;

    @Test
    void embedLinearSoftmaxRoundTrip() throws IOException {
        // Build: Embed (str → R^4) → Linear (4 → 3) → Softmax (3 → 3).
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 42L);
        Embed embed = new Embed(table);
        Linear linear = new Linear(3, 4, true, 7L);
        Softmax softmax = new Softmax();

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("emb", embed)
                .addNode("lin", linear)
                .addNode("sm", softmax)
                .build();

        CompGraphNode nEmb = new CompGraphNode("c-emb", tg.node("emb"));
        CompGraphNode nLin = new CompGraphNode("c-lin", tg.node("lin"));
        CompGraphNode nSm = new CompGraphNode("c-sm", tg.node("sm"));
        nLin.wire(0, new SlotSource(nEmb, tg.edge(tg.node("emb"), tg.node("lin"))));
        nSm.wire(0, new SlotSource(nLin, tg.edge(tg.node("lin"), tg.node("sm"))));
        ComputationGraph cg = new ComputationGraph(List.of(nEmb, nLin, nSm), nSm);

        // Pre-populate the embedding table with three known symbols.
        for (String sym : List.of("alpha", "beta", "gamma")) {
            embed.apply(List.of(new StringValue(sym)));
        }

        // Snapshot in-memory outputs for each symbol.
        cg.bindRoot(nEmb, 0, new StringValue("alpha"));
        double[] yAlpha = ((MatrixValue) cg.execute()).data();
        cg.bindRoot(nEmb, 0, new StringValue("beta"));
        double[] yBeta = ((MatrixValue) cg.execute()).data();
        cg.bindRoot(nEmb, 0, new StringValue("gamma"));
        double[] yGamma = ((MatrixValue) cg.execute()).data();

        // Export.
        Path componentDir = tempDir.resolve("emb-lin-sm.mcc");
        new Exporter().export(cg,
                List.of(new Exporter.RootInput("symbol", nEmb, 0)),
                componentDir);

        // Reload.
        Component component = Component.load(componentDir);

        // Inference must match for every known symbol.
        Value rAlpha = component.infer(Map.of("symbol", new StringValue("alpha")));
        Value rBeta  = component.infer(Map.of("symbol", new StringValue("beta")));
        Value rGamma = component.infer(Map.of("symbol", new StringValue("gamma")));

        assertArrayEquals(yAlpha, ((MatrixValue) rAlpha).data(), 1e-12);
        assertArrayEquals(yBeta,  ((MatrixValue) rBeta).data(),  1e-12);
        assertArrayEquals(yGamma, ((MatrixValue) rGamma).data(), 1e-12);

        // Sanity: softmax output must sum to 1 (proves Softmax wiring + execution survived).
        double s = 0;
        for (double v : ((MatrixValue) rAlpha).data()) s += v;
        assertEquals(1.0, s, 1e-12, "softmax output must sum to 1 after round-trip");
    }

    @Test
    void unknownSymbolStillProducesOutput() throws IOException {
        // Variant: after reload, looking up a symbol not in the original
        // table generates a fresh random embedding via the table's RNG.
        // This proves the embed primitive is still operationally live.
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 42L);
        Embed embed = new Embed(table);
        embed.apply(List.of(new StringValue("known")));

        TransformationGraph tg = new TransformationGraphBuilder().addNode("emb", embed).build();
        CompGraphNode n = new CompGraphNode("c-emb", tg.node("emb"));
        ComputationGraph cg = new ComputationGraph(List.of(n), n);

        Path dir = tempDir.resolve("just-embed.mcc");
        new Exporter().export(cg, List.of(new Exporter.RootInput("sym", n, 0)), dir);

        Component loaded = Component.load(dir);
        // Looking up the known symbol should reproduce the trained vector.
        cg.bindRoot(n, 0, new StringValue("known"));
        MatrixValue originalKnown = (MatrixValue) cg.execute();
        MatrixValue reloadedKnown = (MatrixValue) loaded.infer(Map.of("sym", new StringValue("known")));
        assertArrayEquals(originalKnown.data(), reloadedKnown.data(), 1e-12);

        // Looking up a NEW symbol on the reloaded component must still work
        // (the runtime's table can generate fresh embeddings on demand).
        MatrixValue novelVec = (MatrixValue) loaded.infer(Map.of("sym", new StringValue("novel")));
        assertEquals(4, novelVec.data().length, "novel symbol should still yield a length-4 vector");
    }
}
