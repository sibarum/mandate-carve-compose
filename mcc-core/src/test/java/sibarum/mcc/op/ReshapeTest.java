package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.QuaternionValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ReshapeTest {

    private static final double TOL = 1e-15;

    @Test
    void vectorToTernionRoundTrip() {
        VectorToTernion vt = new VectorToTernion();
        TernionToVector tv = new TernionToVector();
        double[] x = { 1.0, 2.0, 3.0 };
        TernionValue mid = (TernionValue) vt.apply(List.of(new MatrixValue(x)));
        MatrixValue back = (MatrixValue) tv.apply(List.of(mid));
        assertArrayEquals(x, back.data(), TOL);

        // Backward inverts cleanly.
        TernionValue g = new TernionValue(0.5, -0.3, 0.7);
        List<Value> dV = vt.backward(g);
        assertArrayEquals(new double[] { 0.5, -0.3, 0.7 }, ((MatrixValue) dV.getFirst()).data(), TOL);
    }

    @Test
    void vectorToTernionRejectsWrongLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new VectorToTernion().apply(List.of(new MatrixValue(new double[] { 1, 2 }))));
    }

    @Test
    void vectorToQuaternionRoundTrip() {
        VectorToQuaternion vq = new VectorToQuaternion();
        QuaternionToVector qv = new QuaternionToVector();
        double[] x = { 1, 2, 3, 4 };
        QuaternionValue mid = (QuaternionValue) vq.apply(List.of(new MatrixValue(x)));
        MatrixValue back = (MatrixValue) qv.apply(List.of(mid));
        assertArrayEquals(x, back.data(), TOL);
    }

    @Test
    void vectorToQuaternionRejectsWrongLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new VectorToQuaternion().apply(List.of(new MatrixValue(new double[] { 1, 2, 3 }))));
    }

    @Test
    void vectorToTensorRespectsShape() {
        VectorToTensor vt = new VectorToTensor(new int[] { 2, 3 });
        double[] x = { 1, 2, 3, 4, 5, 6 };
        TensorValue out = (TensorValue) vt.apply(List.of(new MatrixValue(x)));
        assertArrayEquals(new int[] { 2, 3 }, out.shape());
        assertArrayEquals(x, out.data(), TOL);

        // Backward: same flat data, shape stays in the gradient input.
        TensorValue g = new TensorValue(new int[] { 2, 3 }, new double[] { 10, 20, 30, 40, 50, 60 });
        List<Value> back = vt.backward(g);
        assertArrayEquals(new double[] { 10, 20, 30, 40, 50, 60 }, ((MatrixValue) back.getFirst()).data(), TOL);
    }

    @Test
    void vectorToTensorRejectsVolumeMismatch() {
        VectorToTensor vt = new VectorToTensor(new int[] { 2, 3 });
        assertThrows(IllegalArgumentException.class,
                () -> vt.apply(List.of(new MatrixValue(new double[5]))));
    }

    @Test
    void tensorToVectorPreservesShapeForBackward() {
        TensorToVector tv = new TensorToVector();
        TensorValue in = new TensorValue(new int[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
        MatrixValue flat = (MatrixValue) tv.apply(List.of(in));
        assertArrayEquals(new double[] { 1, 2, 3, 4, 5, 6 }, flat.data(), TOL);

        double[] g = { 10, 20, 30, 40, 50, 60 };
        TensorValue back = (TensorValue) tv.backward(new MatrixValue(g)).getFirst();
        assertArrayEquals(new int[] { 3, 2 }, back.shape());
        assertArrayEquals(g, back.data(), TOL);
    }

    @Test
    void tensorToVectorBackwardRejectsWrongLength() {
        TensorToVector tv = new TensorToVector();
        tv.apply(List.of(new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 })));
        assertThrows(IllegalArgumentException.class,
                () -> tv.backward(new MatrixValue(new double[] { 1, 2, 3 })));
    }
}
