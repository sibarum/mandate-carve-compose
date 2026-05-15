package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Finite-difference gradient checks for the stateless ops' Differentiable
 * backwards. Loss for each test is {@code L = ⟨gradOut, output⟩} so the
 * analytic backward should match {@code dL/dinput} from finite differences.
 */
class StatelessOpGradientCheckTest {

    private static final double EPS = 1e-6;
    private static final double TOL = 1e-7;

    @Test
    void dotProductBackward() {
        DotProduct op = new DotProduct();
        double[] u = { 0.3, -0.7, 1.1 };
        double[] v = { 0.5, 0.2, -0.4 };
        double gOut = 0.7;
        checkBackward2(op, new MatrixValue(u), new MatrixValue(v), new NumberValue(gOut), u, v);
    }

    @Test
    void magnitudeBackward() {
        Magnitude op = new Magnitude();
        double[] u = { 1.0, -2.0, 3.0 };
        double gOut = 0.5;
        checkBackward1Scalar(op, new MatrixValue(u), new NumberValue(gOut), u);
    }

    @Test
    void normalizeBackward() {
        Normalize op = new Normalize();
        double[] u = { 0.6, 0.8, -0.5, 0.2 };
        double[] gOut = { 0.3, -0.1, 0.4, 0.2 };
        checkBackward1Matrix(op, new MatrixValue(u), new MatrixValue(gOut), u);
    }

    @Test
    void cosineSimilarityBackward() {
        CosineSimilarity op = new CosineSimilarity();
        double[] u = { 0.3, 0.5, -0.7 };
        double[] v = { 0.9, -0.2, 0.4 };
        double gOut = 0.6;
        checkBackward2(op, new MatrixValue(u), new MatrixValue(v), new NumberValue(gOut), u, v);
    }

    @Test
    void crossProduct3Backward() {
        CrossProduct3 op = new CrossProduct3();
        TernionValue u = new TernionValue(0.5, -0.3, 0.7);
        TernionValue v = new TernionValue(-0.2, 0.6, 0.4);
        TernionValue gOut = new TernionValue(0.3, -0.5, 0.1);

        // Forward + analytic backward.
        op.apply(List.of(u, v));
        List<Value> ana = op.backward(gOut);
        TernionValue dU = (TernionValue) ana.get(0);
        TernionValue dV = (TernionValue) ana.get(1);

        // Numerical: perturb each component of u, v.
        double[] uArr = u.toArray();
        double[] vArr = v.toArray();
        double maxErr = 0.0;
        for (int i = 0; i < 3; i++) {
            double orig = uArr[i];
            uArr[i] = orig + EPS;
            double lPlus = lossDotTernion((TernionValue) op.apply(List.of(new TernionValue(uArr), v)), gOut);
            uArr[i] = orig - EPS;
            double lMinus = lossDotTernion((TernionValue) op.apply(List.of(new TernionValue(uArr), v)), gOut);
            uArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dU.toArray()[i]));
        }
        for (int i = 0; i < 3; i++) {
            double orig = vArr[i];
            vArr[i] = orig + EPS;
            double lPlus = lossDotTernion((TernionValue) op.apply(List.of(u, new TernionValue(vArr))), gOut);
            vArr[i] = orig - EPS;
            double lMinus = lossDotTernion((TernionValue) op.apply(List.of(u, new TernionValue(vArr))), gOut);
            vArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dV.toArray()[i]));
        }
        assertTrue(maxErr < TOL, "CrossProduct3 max err " + maxErr + " > " + TOL);
    }

    @Test
    void concatBackward() {
        Concat op = new Concat();
        double[] u = { 1, 2, 3 };
        double[] v = { 4, 5 };
        double[] gOut = { 0.1, 0.2, 0.3, 0.4, 0.5 };
        op.apply(List.of(new MatrixValue(u), new MatrixValue(v)));
        List<Value> ana = op.backward(new MatrixValue(gOut));
        // dU should be gOut[0:3], dV should be gOut[3:5].
        assertArrayEquals(new double[] { 0.1, 0.2, 0.3 }, ((MatrixValue) ana.get(0)).data(), 1e-15);
        assertArrayEquals(new double[] { 0.4, 0.5 }, ((MatrixValue) ana.get(1)).data(), 1e-15);
    }

    @Test
    void sliceBackward() {
        Slice op = new Slice(1, 4);
        double[] x = { 0, 1, 2, 3, 4, 5 };
        double[] gOut = { 10, 20, 30 };
        op.apply(List.of(new MatrixValue(x)));
        List<Value> ana = op.backward(new MatrixValue(gOut));
        // Zero-padded back into a length-6 vector at indices [1, 4).
        assertArrayEquals(new double[] { 0, 10, 20, 30, 0, 0 }, ((MatrixValue) ana.get(0)).data(), 1e-15);
    }

    // ---- helpers ----

    /** For binary MatrixValue → NumberValue ops with scalar gradOut. */
    private static void checkBackward2(Differentiable op, MatrixValue u, MatrixValue v,
                                        NumberValue gOut, double[] uArr, double[] vArr) {
        op.apply(List.of(u, v));
        List<Value> ana = op.backward(gOut);
        double[] dU = ((MatrixValue) ana.get(0)).data();
        double[] dV = ((MatrixValue) ana.get(1)).data();

        double maxErr = 0.0;
        for (int i = 0; i < uArr.length; i++) {
            double orig = uArr[i];
            uArr[i] = orig + EPS;
            double lPlus = scalarLoss(op, new MatrixValue(uArr), v, gOut);
            uArr[i] = orig - EPS;
            double lMinus = scalarLoss(op, new MatrixValue(uArr), v, gOut);
            uArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dU[i]));
        }
        for (int i = 0; i < vArr.length; i++) {
            double orig = vArr[i];
            vArr[i] = orig + EPS;
            double lPlus = scalarLoss(op, u, new MatrixValue(vArr), gOut);
            vArr[i] = orig - EPS;
            double lMinus = scalarLoss(op, u, new MatrixValue(vArr), gOut);
            vArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dV[i]));
        }
        assertTrue(maxErr < TOL, op.name() + " max err " + maxErr + " > " + TOL);
    }

    /** For unary MatrixValue → NumberValue ops. */
    private static void checkBackward1Scalar(Differentiable op, MatrixValue u,
                                              NumberValue gOut, double[] uArr) {
        op.apply(List.of(u));
        List<Value> ana = op.backward(gOut);
        double[] dU = ((MatrixValue) ana.get(0)).data();

        double maxErr = 0.0;
        for (int i = 0; i < uArr.length; i++) {
            double orig = uArr[i];
            uArr[i] = orig + EPS;
            double lPlus = scalarLoss(op, new MatrixValue(uArr), null, gOut);
            uArr[i] = orig - EPS;
            double lMinus = scalarLoss(op, new MatrixValue(uArr), null, gOut);
            uArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dU[i]));
        }
        assertTrue(maxErr < TOL, op.name() + " max err " + maxErr + " > " + TOL);
    }

    /** For unary MatrixValue → MatrixValue ops with matrix gradOut. */
    private static void checkBackward1Matrix(Differentiable op, MatrixValue u,
                                              MatrixValue gOut, double[] uArr) {
        op.apply(List.of(u));
        List<Value> ana = op.backward(gOut);
        double[] dU = ((MatrixValue) ana.get(0)).data();

        double maxErr = 0.0;
        for (int i = 0; i < uArr.length; i++) {
            double orig = uArr[i];
            uArr[i] = orig + EPS;
            double lPlus = matrixLoss(op, new MatrixValue(uArr), gOut);
            uArr[i] = orig - EPS;
            double lMinus = matrixLoss(op, new MatrixValue(uArr), gOut);
            uArr[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            maxErr = Math.max(maxErr, Math.abs(numeric - dU[i]));
        }
        assertTrue(maxErr < TOL, op.name() + " max err " + maxErr + " > " + TOL);
    }

    private static double scalarLoss(Differentiable op, MatrixValue u, MatrixValue v, NumberValue gOut) {
        Value out = v == null ? op.apply(List.of(u)) : op.apply(List.of(u, v));
        return ((NumberValue) out).n() * gOut.n();
    }

    private static double matrixLoss(Differentiable op, MatrixValue u, MatrixValue gOut) {
        MatrixValue out = (MatrixValue) op.apply(List.of(u));
        double s = 0;
        for (int i = 0; i < out.data().length; i++) s += out.data()[i] * gOut.data()[i];
        return s;
    }

    private static double lossDotTernion(TernionValue out, TernionValue g) {
        return out.x() * g.x() + out.y() * g.y() + out.z() * g.z();
    }
}
