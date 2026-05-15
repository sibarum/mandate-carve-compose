package sibarum.mcc.embedding;

import org.junit.jupiter.api.Test;
import sibarum.mcc.primitive.Parameterized.NamedTensor;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class EmbedParameterizedTest {

    private static final double TOL = 1e-15;

    @Test
    void configCarriesInsertionOrderedSymbols() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(4, 42L);
        Embed e = new Embed(t);
        e.apply(List.of(new StringValue("foo")));
        e.apply(List.of(new StringValue("bar")));
        e.apply(List.of(new StringValue("baz")));

        Map<String, Object> cfg = e.config();
        assertEquals(4, cfg.get("dim"));
        @SuppressWarnings("unchecked")
        List<String> symbols = (List<String>) cfg.get("symbols");
        assertEquals(List.of("foo", "bar", "baz"), symbols,
                "Embed config must preserve insertion order");
    }

    @Test
    void parametersIsSingleConcatenatedTensor() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(4, 42L);
        Embed e = new Embed(t);
        e.apply(List.of(new StringValue("a")));
        e.apply(List.of(new StringValue("b")));

        List<NamedTensor> ps = e.parameters();
        assertEquals(1, ps.size());
        NamedTensor nt = ps.getFirst();
        assertEquals("embeddings", nt.name());
        assertArrayEquals(new int[] { 2, 4 }, nt.shape());
        assertEquals(8, nt.data().length);
    }

    @Test
    void roundTripPreservesEmbeddings() {
        SymbolEmbeddingTable src = new SymbolEmbeddingTable(4, 42L);
        Embed e = new Embed(src);
        for (String s : List.of("x", "y", "z")) e.apply(List.of(new StringValue(s)));

        // Snapshot source state.
        Map<String, Object> cfg = e.config();
        Map<String, NamedTensor> snapshot = new HashMap<>();
        for (NamedTensor t : e.parameters()) snapshot.put(t.name(), t);

        // Rebuild a fresh table with the same symbol set in the same order.
        @SuppressWarnings("unchecked")
        List<String> symbols = (List<String>) cfg.get("symbols");
        SymbolEmbeddingTable dst = new SymbolEmbeddingTable((int) cfg.get("dim"), 999L);
        for (String sym : symbols) dst.embed(sym);  // pre-populate in order
        Embed e2 = new Embed(dst);
        e2.loadParameters(snapshot);

        // Outputs must match for every symbol.
        for (String sym : symbols) {
            MatrixValue srcVec = (MatrixValue) e.apply(List.of(new StringValue(sym)));
            MatrixValue dstVec = (MatrixValue) e2.apply(List.of(new StringValue(sym)));
            assertArrayEquals(srcVec.data(), dstVec.data(), TOL,
                    "embedding for '" + sym + "' did not round-trip");
        }
    }

    @Test
    void wrongShapeLoadThrowsWithHelpfulMessage() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(4, 42L);
        Embed e = new Embed(t);
        e.apply(List.of(new StringValue("a")));

        Map<String, NamedTensor> bogus = new HashMap<>();
        // Table has 1 symbol × 4 dim; supply 2×4 instead.
        bogus.put("embeddings", new NamedTensor("embeddings", new int[] { 2, 4 }, new double[8]));
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> e.loadParameters(bogus));
        assertTrue(ex.getMessage().contains("registry must pre-populate"),
                "error should hint at the pre-population requirement, got: " + ex.getMessage());
    }

    @Test
    void emptyTableRoundTrips() {
        SymbolEmbeddingTable src = new SymbolEmbeddingTable(4, 42L);
        Embed e = new Embed(src);
        List<NamedTensor> ps = e.parameters();
        assertEquals(1, ps.size());
        assertArrayEquals(new int[] { 0, 4 }, ps.getFirst().shape());

        // Load into a fresh (also empty) table.
        SymbolEmbeddingTable dst = new SymbolEmbeddingTable(4, 999L);
        Embed e2 = new Embed(dst);
        Map<String, NamedTensor> snap = new HashMap<>();
        for (NamedTensor t : ps) snap.put(t.name(), t);
        e2.loadParameters(snap);
        assertEquals(0, dst.size());
    }
}
