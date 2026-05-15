package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.Value;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MatrixOpsGradientCheckTest {

    private static final double EPS = 1e-6;
    private static final double TOL = 1e-8;

    @Test
    void matMulForwardMatchesManualComputation() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A·B = [[19, 22], [43, 50]]
        TensorValue a = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        TensorValue b = new TensorValue(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
        TensorValue c = (TensorValue) new MatMul().apply(List.of(a, b));
        assertArrayEquals(new int[] { 2, 2 }, c.shape());
        assertArrayEquals(new double[] { 19, 22, 43, 50 }, c.data(), 1e-12);
    }

    @Test
    void matMulBackwardMatchesFiniteDifferences() {
        MatMul op = new MatMul();
        double[] aD = { 0.1, -0.3, 0.7, 0.2, -0.5, 0.6 };  // 2x3
        double[] bD = { 0.4, -0.1, 0.5, -0.2, 0.3, 0.6 };  // 3x2
        double[] gD = { 0.7, -0.2, 0.1, 0.3 };             // 2x2
        TensorValue a = new TensorValue(new int[] { 2, 3 }, aD);
        TensorValue b = new TensorValue(new int[] { 3, 2 }, bD);
        TensorValue g = new TensorValue(new int[] { 2, 2 }, gD);

        op.apply(List.of(a, b));
        List<Value> grads = op.backward(g);
        TensorValue dA = (TensorValue) grads.get(0);
        TensorValue dB = (TensorValue) grads.get(1);

        double maxErr = 0;
        // Check ∂L/∂A.
        for (int i = 0; i < aD.length; i++) {
            double orig = aD[i];
            aD[i] = orig + EPS;
            double lPlus = lossDotTensor(
                    (TensorValue) op.apply(List.of(new TensorValue(new int[] { 2, 3 }, aD), b)), g);
            aD[i] = orig - EPS;
            double lMinus = lossDotTensor(
                    (TensorValue) op.apply(List.of(new TensorValue(new int[] { 2, 3 }, aD), b)), g);
            aD[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dA.data()[i]));
        }
        // Check ∂L/∂B.
        for (int i = 0; i < bD.length; i++) {
            double orig = bD[i];
            bD[i] = orig + EPS;
            double lPlus = lossDotTensor(
                    (TensorValue) op.apply(List.of(a, new TensorValue(new int[] { 3, 2 }, bD))), g);
            bD[i] = orig - EPS;
            double lMinus = lossDotTensor(
                    (TensorValue) op.apply(List.of(a, new TensorValue(new int[] { 3, 2 }, bD))), g);
            bD[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dB.data()[i]));
        }
        assertTrue(maxErr < TOL, "MatMul max err " + maxErr + " > " + TOL);
    }

    @Test
    void matMulRejectsRankAndShapeMismatch() {
        MatMul op = new MatMul();
        TensorValue rank1 = new TensorValue(new int[] { 3 }, new double[] { 1, 2, 3 });
        TensorValue rank2 = new TensorValue(new int[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
        assertThrows(IllegalArgumentException.class, () -> op.apply(List.of(rank1, rank2)));

        TensorValue bad = new TensorValue(new int[] { 5, 2 }, new double[10]);
        assertThrows(IllegalArgumentException.class,
                () -> op.apply(List.of(new TensorValue(new int[] { 2, 3 }, new double[6]), bad)));
    }

    @Test
    void matVecMulForwardMatchesManualComputation() {
        // A = [[1, 2, 3], [4, 5, 6]], v = [7, 8, 9]
        // y = [50, 122]
        TensorValue a = new TensorValue(new int[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
        MatrixValue v = new MatrixValue(new double[] { 7, 8, 9 });
        MatrixValue y = (MatrixValue) new MatVecMul().apply(List.of(a, v));
        assertArrayEquals(new double[] { 50, 122 }, y.data(), 1e-12);
    }

    @Test
    void matVecMulBackwardMatchesFiniteDifferences() {
        MatVecMul op = new MatVecMul();
        double[] aD = { 0.5, -0.3, 0.7, 0.2, 0.6, -0.4 };  // 2x3
        double[] vD = { 0.4, -0.1, 0.8 };
        double[] gD = { 0.7, -0.5 };
        TensorValue a = new TensorValue(new int[] { 2, 3 }, aD);
        MatrixValue v = new MatrixValue(vD);
        MatrixValue g = new MatrixValue(gD);

        op.apply(List.of(a, v));
        List<Value> grads = op.backward(g);
        TensorValue dA = (TensorValue) grads.get(0);
        MatrixValue dv = (MatrixValue) grads.get(1);

        double maxErr = 0;
        for (int i = 0; i < aD.length; i++) {
            double orig = aD[i];
            aD[i] = orig + EPS;
            double lPlus = lossDot(
                    (MatrixValue) op.apply(List.of(new TensorValue(new int[] { 2, 3 }, aD), v)), gD);
            aD[i] = orig - EPS;
            double lMinus = lossDot(
                    (MatrixValue) op.apply(List.of(new TensorValue(new int[] { 2, 3 }, aD), v)), gD);
            aD[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dA.data()[i]));
        }
        for (int j = 0; j < vD.length; j++) {
            double orig = vD[j];
            vD[j] = orig + EPS;
            double lPlus = lossDot(
                    (MatrixValue) op.apply(List.of(a, new MatrixValue(vD))), gD);
            vD[j] = orig - EPS;
            double lMinus = lossDot(
                    (MatrixValue) op.apply(List.of(a, new MatrixValue(vD))), gD);
            vD[j] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dv.data()[j]));
        }
        assertTrue(maxErr < TOL, "MatVecMul max err " + maxErr + " > " + TOL);
    }

    private static double lossDotTensor(TensorValue out, TensorValue g) {
        double s = 0;
        for (int i = 0; i < out.data().length; i++) s += out.data()[i] * g.data()[i];
        return s;
    }

    private static double lossDot(MatrixValue out, double[] g) {
        double s = 0;
        for (int i = 0; i < out.data().length; i++) s += out.data()[i] * g[i];
        return s;
    }
}
