package sibarum.mcc.carving;

import org.junit.jupiter.api.Test;
import sibarum.mcc.embedding.Embed;
import sibarum.mcc.embedding.Lookup;
import sibarum.mcc.embedding.SymbolEmbeddingTable;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.mandate.Mandate;
import sibarum.mcc.mandate.MandateSet;
import sibarum.mcc.op.Terminal;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BackwardChainingCarverTest {

    @Test
    void carvesDirectIdentityFromRootToTerminal() {
        // Smallest possible carving: root input is a MatrixValue,
        // result mandate is the same MatrixValue. Carver should wire
        // Terminal slot directly to the root.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("term", new Terminal(ValueType.MATRIX))
                .build();
        MatrixValue x = new MatrixValue(new double[] { 1.0, 2.0, 3.0 });
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(x, 1e-9, 10)
        ));

        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        CarvingResult result = carver.carve(tg, mandates, x);

        assertNotNull(result, "carver returned null on trivial identity carving");
        assertEquals(1, result.graph().nodes().size(),
                "expected only the terminal node in the carving");
        Value out = result.graph().execute();
        assertArrayEquals(x.data(), ((MatrixValue) out).data(), 1e-12);
        assertEquals(1, result.rootBindings().size(), "expected exactly one root binding");
    }

    @Test
    void carvesEmbedLookupChain() {
        // Substrate: Embed (str → matrix), Lookup (matrix → str),
        // Terminal-matrix. Root is a string "foo". Mandate is the
        // matrix that Embed("foo") produces. The carver should:
        //   Terminal ← Embed ← root.
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 42L);
        Embed embed = new Embed(table);
        Lookup lookup = new Lookup(table);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("emb", embed)
                .addNode("look", lookup)
                .addNode("term", new Terminal(ValueType.MATRIX))
                .build();

        StringValue root = new StringValue("foo");
        // Materialize the embedding so the mandate has a real target value.
        MatrixValue targetVec = new MatrixValue(table.embed("foo"));
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(targetVec, 1e-9, 10)
        ));

        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        CarvingResult result = carver.carve(tg, mandates, root);

        assertNotNull(result, "expected a successful carving");
        // Terminal + Embed = 2 nodes.
        assertEquals(2, result.graph().nodes().size());
        Value out = result.graph().execute();
        assertArrayEquals(targetVec.data(), ((MatrixValue) out).data(), 1e-12);
    }

    @Test
    void returnsNullWhenNoTerminalForType() {
        // No Terminal of the result type in the substrate — carver must
        // gracefully return null.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("term-string", new Terminal(ValueType.STRING))
                .build();
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new MatrixValue(new double[] { 1.0 }), 0.0, 10)
        ));
        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        assertNull(carver.carve(tg, mandates, new MatrixValue(new double[] { 1.0 })));
    }

    @Test
    void inversionContextProvidesAnchorAndReachable() {
        // Indirect test: a Lookup primitive's inversion uses the table's
        // exact stored vector — verifies that the carver's pre-population
        // is correctly observable through the InversionContext indirectly
        // (Lookup doesn't actually use the context, but Embed does).
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 7L);
        Embed embed = new Embed(table);
        embed.apply(List.of(new StringValue("a")));
        embed.apply(List.of(new StringValue("b")));

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("emb", embed)
                .addNode("term", new Terminal(ValueType.MATRIX))
                .build();

        // Target: a vector close to "a"'s embedding.
        MatrixValue target = new MatrixValue(table.embed("a"));
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(target, 1e-9, 10)
        ));
        // Root is "a" (so Embed("a") == target after the root chain).
        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        CarvingResult result = carver.carve(tg, mandates, new StringValue("a"));
        assertNotNull(result);
        assertArrayEquals(target.data(),
                ((MatrixValue) result.graph().execute()).data(), 1e-12);
    }
}
