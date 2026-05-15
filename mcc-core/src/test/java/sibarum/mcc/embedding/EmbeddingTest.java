package sibarum.mcc.embedding;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingTest {

    private static final double TOL = 1e-12;

    @Test
    void embedReturnsSameVectorForSameSymbol() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(8, 42L);
        Embed e = new Embed(t);
        MatrixValue v1 = (MatrixValue) e.apply(List.of(new StringValue("hello")));
        MatrixValue v2 = (MatrixValue) e.apply(List.of(new StringValue("hello")));
        assertArrayEquals(v1.data(), v2.data(), TOL);
    }

    @Test
    void embedReturnsDifferentVectorsForDifferentSymbols() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(8, 42L);
        Embed e = new Embed(t);
        MatrixValue v1 = (MatrixValue) e.apply(List.of(new StringValue("a")));
        MatrixValue v2 = (MatrixValue) e.apply(List.of(new StringValue("b")));
        boolean different = false;
        for (int i = 0; i < v1.data().length; i++) {
            if (Math.abs(v1.data()[i] - v2.data()[i]) > 1e-9) {
                different = true;
                break;
            }
        }
        assertTrue(different, "Distinct symbols should get distinct random embeddings");
    }

    @Test
    void lookupRecoversInsertedSymbol() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(8, 42L);
        Embed e = new Embed(t);
        Lookup l = new Lookup(t);
        MatrixValue v = (MatrixValue) e.apply(List.of(new StringValue("foo")));
        e.apply(List.of(new StringValue("bar")));  // populate a second
        StringValue hit = (StringValue) l.apply(List.of(v));
        assertEquals("foo", hit.s());
    }

    @Test
    void embedTrainableIdentityIsTheTable() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(8, 42L);
        Embed a = new Embed(t);
        Embed b = new Embed(t);
        assertSame(t, a.trainableIdentity());
        assertSame(a.trainableIdentity(), b.trainableIdentity(),
                "Two Embeds sharing a table must share trainableIdentity for trainer dedup");
    }

    @Test
    void embedBackwardStepMovesTowardTarget() {
        SymbolEmbeddingTable t = new SymbolEmbeddingTable(4, 0L);
        Embed e = new Embed(t);
        StringValue sym = new StringValue("x");
        MatrixValue start = (MatrixValue) e.apply(List.of(sym));
        double[] target = { 1.0, 1.0, 1.0, 1.0 };

        double initialL2 = distance(start.data(), target);

        // Gradient-flow shape: gradOut = output - target under MSE.
        double[] gradOut = new double[target.length];
        for (int i = 0; i < target.length; i++) gradOut[i] = start.data()[i] - target[i];
        e.backward(new MatrixValue(gradOut));
        e.step(0.5);  // big step, should clearly move toward target

        MatrixValue after = (MatrixValue) e.apply(List.of(sym));
        double afterL2 = distance(after.data(), target);
        assertTrue(afterL2 < initialL2,
                "embedding did not move toward target: before=" + initialL2 + " after=" + afterL2);
    }

    private static double distance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }
}
