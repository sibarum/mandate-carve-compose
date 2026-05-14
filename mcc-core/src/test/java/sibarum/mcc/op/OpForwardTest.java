package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class OpForwardTest {

    private static final double TOL = 1e-12;

    @Test
    void addComponentwise() {
        Value out = new Add().apply(List.of(
                new MatrixValue(new double[] { 1, 2, 3 }),
                new MatrixValue(new double[] { 4, 5, 6 })));
        assertArrayEquals(new double[] { 5, 7, 9 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void addTotalArithmeticInfCancellation() {
        Value out = new Add().apply(List.of(
                new MatrixValue(new double[] { Double.POSITIVE_INFINITY, 1.0 }),
                new MatrixValue(new double[] { Double.NEGATIVE_INFINITY, 1.0 })));
        double[] d = ((MatrixValue) out).data();
        assertEquals(0.0, d[0], TOL);
        assertEquals(2.0, d[1], TOL);
    }

    @Test
    void subComponentwise() {
        Value out = new Sub().apply(List.of(
                new MatrixValue(new double[] { 5, 5 }),
                new MatrixValue(new double[] { 1, 2 })));
        assertArrayEquals(new double[] { 4, 3 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void mulHadamard() {
        Value out = new Mul().apply(List.of(
                new MatrixValue(new double[] { 2, 3, 4 }),
                new MatrixValue(new double[] { 5, 6, 7 })));
        assertArrayEquals(new double[] { 10, 18, 28 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void dimMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () -> new Add().apply(List.of(
                new MatrixValue(new double[] { 1 }), new MatrixValue(new double[] { 1, 2 }))));
    }

    @Test
    void dotProduct() {
        Value out = new DotProduct().apply(List.of(
                new MatrixValue(new double[] { 1, 2, 3 }),
                new MatrixValue(new double[] { 4, 5, 6 })));
        assertEquals(32.0, ((NumberValue) out).n(), TOL);
    }

    @Test
    void magnitudeL2() {
        Value out = new Magnitude().apply(List.of(new MatrixValue(new double[] { 3, 4 })));
        assertEquals(5.0, ((NumberValue) out).n(), TOL);
    }

    @Test
    void normalizeUnit() {
        Value out = new Normalize().apply(List.of(new MatrixValue(new double[] { 3, 4 })));
        double[] d = ((MatrixValue) out).data();
        assertEquals(0.6, d[0], TOL);
        assertEquals(0.8, d[1], TOL);
    }

    @Test
    void normalizeZeroVector() {
        Value out = new Normalize().apply(List.of(new MatrixValue(new double[] { 0, 0, 0 })));
        assertArrayEquals(new double[] { 0, 0, 0 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void softmaxSumsToOne() {
        Value out = new Softmax().apply(List.of(new MatrixValue(new double[] { 1, 2, 3 })));
        double[] d = ((MatrixValue) out).data();
        double sum = 0;
        for (double v : d) sum += v;
        assertEquals(1.0, sum, TOL);
        assertTrue(d[2] > d[1] && d[1] > d[0]);
    }

    @Test
    void softmaxStableUnderLargeLogits() {
        Value out = new Softmax().apply(List.of(new MatrixValue(new double[] { 1000, 1001, 999 })));
        double[] d = ((MatrixValue) out).data();
        double sum = 0;
        for (double v : d) sum += v;
        assertEquals(1.0, sum, TOL);
        assertFalse(Double.isNaN(d[0]));
    }

    @Test
    void crossProduct3StandardBasis() {
        Value out = new CrossProduct3().apply(List.of(
                new TernionValue(1, 0, 0), new TernionValue(0, 1, 0)));
        TernionValue t = (TernionValue) out;
        assertEquals(0.0, t.x(), TOL);
        assertEquals(0.0, t.y(), TOL);
        assertEquals(1.0, t.z(), TOL);
    }

    @Test
    void crossProduct3Antisymmetric() {
        TernionValue u = new TernionValue(1, 2, 3);
        TernionValue v = new TernionValue(4, 5, 6);
        TernionValue uxv = (TernionValue) new CrossProduct3().apply(List.of(u, v));
        TernionValue vxu = (TernionValue) new CrossProduct3().apply(List.of(v, u));
        assertEquals(uxv.x(), -vxu.x(), TOL);
        assertEquals(uxv.y(), -vxu.y(), TOL);
        assertEquals(uxv.z(), -vxu.z(), TOL);
    }

    @Test
    void relu() {
        Value out = new Relu().apply(List.of(new MatrixValue(new double[] { -1, 0, 2, -0.5, 3 })));
        assertArrayEquals(new double[] { 0, 0, 2, 0, 3 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void sigmoidMonotonic() {
        Value out = new Sigmoid().apply(List.of(new MatrixValue(new double[] { -10, 0, 10 })));
        double[] d = ((MatrixValue) out).data();
        assertTrue(d[0] < 0.001);
        assertEquals(0.5, d[1], TOL);
        assertTrue(d[2] > 0.999);
    }

    @Test
    void sigmoidExtremesNoNaN() {
        Value out = new Sigmoid().apply(List.of(
                new MatrixValue(new double[] { -1e6, 1e6 })));
        double[] d = ((MatrixValue) out).data();
        assertFalse(Double.isNaN(d[0]));
        assertFalse(Double.isNaN(d[1]));
    }

    @Test
    void tanh() {
        Value out = new Tanh().apply(List.of(new MatrixValue(new double[] { 0, 100, -100 })));
        double[] d = ((MatrixValue) out).data();
        assertEquals(0.0, d[0], TOL);
        assertEquals(1.0, d[1], TOL);
        assertEquals(-1.0, d[2], TOL);
    }

    @Test
    void concatEndToEnd() {
        Value out = new Concat().apply(List.of(
                new MatrixValue(new double[] { 1, 2 }),
                new MatrixValue(new double[] { 3, 4, 5 })));
        assertArrayEquals(new double[] { 1, 2, 3, 4, 5 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void sliceContiguous() {
        Value out = new Slice(1, 4).apply(List.of(
                new MatrixValue(new double[] { 0, 1, 2, 3, 4, 5 })));
        assertArrayEquals(new double[] { 1, 2, 3 }, ((MatrixValue) out).data(), TOL);
    }

    @Test
    void cosineSimilarityIdenticalIsOne() {
        Value out = new CosineSimilarity().apply(List.of(
                new MatrixValue(new double[] { 1, 2, 3 }),
                new MatrixValue(new double[] { 2, 4, 6 })));
        assertEquals(1.0, ((NumberValue) out).n(), TOL);
    }

    @Test
    void cosineSimilarityOrthogonalIsZero() {
        Value out = new CosineSimilarity().apply(List.of(
                new MatrixValue(new double[] { 1, 0 }),
                new MatrixValue(new double[] { 0, 1 })));
        assertEquals(0.0, ((NumberValue) out).n(), TOL);
    }

    @Test
    void similarityGateOrthogonalProducesZero() {
        Value out = new SimilarityGate().apply(List.of(
                new MatrixValue(new double[] { 1, 0 }),
                new MatrixValue(new double[] { 0, 1 })));
        assertArrayEquals(new double[] { 0, 0 }, ((MatrixValue) out).data(), TOL);
    }
}
