package sibarum.mcc.op.block;

import org.junit.jupiter.api.Test;
import sibarum.mcc.primitive.Parameterized.NamedTensor;
import sibarum.mcc.value.MatrixValue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TransformerBlockParameterizedTest {

    private static final double TOL = 1e-15;

    @Test
    void parametersCoverAllTenTensors() {
        TransformerBlock tb = new TransformerBlock(2, 3, 4, 8, 2, 7L);
        List<NamedTensor> ps = tb.parameters();
        assertEquals(10, ps.size(),
                "TransformerBlock should expose Win, Wq, Wk, Wv, W1, b1, W2, b2, Wo, bo");
        Map<String, NamedTensor> byName = new HashMap<>();
        for (NamedTensor t : ps) byName.put(t.name(), t);
        for (String n : List.of("Win", "Wq", "Wk", "Wv", "W1", "b1", "W2", "b2", "Wo", "bo")) {
            assertTrue(byName.containsKey(n), "missing tensor: " + n);
        }
    }

    @Test
    void roundTripPreservesInferenceOutput() {
        TransformerBlock src = new TransformerBlock(2, 3, 4, 8, 2, 7L);
        double[] x = { 0.1, -0.2, 0.3, 0.4, -0.5, 0.6 };
        double[] y0 = ((MatrixValue) src.apply(List.of(new MatrixValue(x)))).data();

        // Fresh instance, same shape, different seed (will have different weights).
        TransformerBlock dst = new TransformerBlock(2, 3, 4, 8, 2, 999L);
        double[] yDiff = ((MatrixValue) dst.apply(List.of(new MatrixValue(x)))).data();
        boolean differBefore = false;
        for (int i = 0; i < y0.length; i++) {
            if (Math.abs(y0[i] - yDiff[i]) > 1e-9) { differBefore = true; break; }
        }
        assertTrue(differBefore, "different seeds must produce different outputs pre-load");

        Map<String, NamedTensor> snapshot = new HashMap<>();
        for (NamedTensor t : src.parameters()) snapshot.put(t.name(), t);
        dst.loadParameters(snapshot);

        double[] yAfter = ((MatrixValue) dst.apply(List.of(new MatrixValue(x)))).data();
        assertArrayEquals(y0, yAfter, TOL, "loadParameters must reproduce source's output exactly");
    }

    @Test
    void shapeMismatchOnLoadThrows() {
        TransformerBlock tb = new TransformerBlock(2, 3, 4, 8, 2, 7L);
        Map<String, NamedTensor> bogus = new HashMap<>();
        for (NamedTensor t : tb.parameters()) bogus.put(t.name(), t);
        bogus.put("Win", new NamedTensor("Win", new int[] { 999, 4 }, new double[999 * 4]));
        assertThrows(IllegalArgumentException.class, () -> tb.loadParameters(bogus));
    }

    @Test
    void configRoundTripsThroughRegistry() {
        TransformerBlock tb = new TransformerBlock(2, 3, 4, 8, 2, 7L);
        Map<String, Object> cfg = tb.config();
        assertEquals(2, cfg.get("seqLen"));
        assertEquals(3, cfg.get("dIn"));
        assertEquals(4, cfg.get("dModel"));
        assertEquals(8, cfg.get("dFf"));
        assertEquals(2, cfg.get("dOut"));
    }
}
