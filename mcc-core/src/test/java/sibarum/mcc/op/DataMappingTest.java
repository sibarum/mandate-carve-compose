package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.primitive.Parameterized.NamedTensor;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DataMappingTest {

    private static final double TOL = 1e-12;

    @Test
    void intToVectorReturnsRowForIndex() {
        IntToVector v = new IntToVector(4, 3, 0L);
        MatrixValue r0 = (MatrixValue) v.apply(List.of(new NumberValue(0)));
        MatrixValue r1 = (MatrixValue) v.apply(List.of(new NumberValue(1)));
        // Different rows must (probabilistically) differ.
        boolean differ = false;
        for (int i = 0; i < 3; i++) if (Math.abs(r0.data()[i] - r1.data()[i]) > 1e-9) { differ = true; break; }
        assertTrue(differ);
        assertEquals(3, r0.data().length);
    }

    @Test
    void intToVectorRejectsOutOfRange() {
        IntToVector v = new IntToVector(4, 3, 0L);
        assertThrows(IllegalArgumentException.class,
                () -> v.apply(List.of(new NumberValue(-1))));
        assertThrows(IllegalArgumentException.class,
                () -> v.apply(List.of(new NumberValue(4))));
    }

    @Test
    void intToVectorRoundsToNearestInt() {
        IntToVector v = new IntToVector(4, 3, 0L);
        MatrixValue r2 = (MatrixValue) v.apply(List.of(new NumberValue(2.0)));
        MatrixValue r2b = (MatrixValue) v.apply(List.of(new NumberValue(1.9)));
        assertArrayEquals(r2.data(), r2b.data(), TOL, "1.9 should round to 2");
    }

    @Test
    void intToVectorBackwardOnlyAffectsIndexedRow() {
        IntToVector v = new IntToVector(4, 3, 0L);
        double[] beforeRow0 = ((MatrixValue) v.apply(List.of(new NumberValue(0)))).data().clone();
        double[] beforeRow2 = ((MatrixValue) v.apply(List.of(new NumberValue(2)))).data().clone();

        // Apply on row 2, backward, step. Row 0 must be unchanged; row 2 must move.
        v.apply(List.of(new NumberValue(2)));
        v.backward(new MatrixValue(new double[] { 1.0, 1.0, 1.0 }));
        v.step(0.1);

        double[] afterRow0 = ((MatrixValue) v.apply(List.of(new NumberValue(0)))).data();
        double[] afterRow2 = ((MatrixValue) v.apply(List.of(new NumberValue(2)))).data();

        assertArrayEquals(beforeRow0, afterRow0, TOL, "row 0 must not change");
        for (int i = 0; i < 3; i++) {
            assertEquals(beforeRow2[i] - 0.1, afterRow2[i], TOL, "row 2 at index " + i);
        }
    }

    @Test
    void intToVectorBackwardReturnsNullForDiscreteInput() {
        IntToVector v = new IntToVector(4, 3, 0L);
        v.apply(List.of(new NumberValue(1)));
        List<Value> grads = v.backward(new MatrixValue(new double[] { 1.0, 2.0, 3.0 }));
        assertEquals(1, grads.size());
        assertNull(grads.getFirst(), "Integer index has no continuous gradient");
    }

    @Test
    void intToVectorParameterRoundTrip() {
        IntToVector src = new IntToVector(3, 2, 42L);
        double[] before = ((MatrixValue) src.apply(List.of(new NumberValue(1)))).data().clone();

        IntToVector dst = new IntToVector(3, 2, 999L);
        Map<String, NamedTensor> snap = new HashMap<>();
        for (NamedTensor t : src.parameters()) snap.put(t.name(), t);
        dst.loadParameters(snap);

        double[] after = ((MatrixValue) dst.apply(List.of(new NumberValue(1)))).data();
        assertArrayEquals(before, after, TOL);
    }

    @Test
    void vectorToIntArgmax() {
        VectorToInt op = new VectorToInt();
        Value r = op.apply(List.of(new MatrixValue(new double[] { 0.1, 0.9, 0.5, -0.2 })));
        assertEquals(1, ((NumberValue) r).n(), TOL);
    }

    @Test
    void vectorToIntTiesGoToLowestIndex() {
        VectorToInt op = new VectorToInt();
        Value r = op.apply(List.of(new MatrixValue(new double[] { 0.5, 0.5, 0.5 })));
        assertEquals(0, ((NumberValue) r).n(), TOL);
    }

    @Test
    void vectorToIntBackwardIsNull() {
        VectorToInt op = new VectorToInt();
        op.apply(List.of(new MatrixValue(new double[] { 0.1, 0.9 })));
        List<Value> grads = op.backward(new NumberValue(1.0));
        assertEquals(1, grads.size());
        assertNull(grads.getFirst(), "argmax has no continuous gradient");
    }

    @Test
    void vectorToIntRejectsEmpty() {
        VectorToInt op = new VectorToInt();
        assertThrows(IllegalArgumentException.class,
                () -> op.apply(List.of(new MatrixValue(new double[0]))));
    }
}
