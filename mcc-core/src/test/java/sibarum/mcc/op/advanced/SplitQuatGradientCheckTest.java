package sibarum.mcc.op.advanced;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Finite-difference gradient check for {@link SplitQuat}'s {@code mul}
 * and {@code sandwich} backward formulas. Compares analytic gradients
 * to numerical derivatives of a scalar loss
 * {@code L = ⟨dC, c⟩} or {@code L = ⟨dY, y⟩} (so that {@code dL/dc =
 * dC}, {@code dL/dy = dY} by construction).
 */
class SplitQuatGradientCheckTest {

    private static final double EPS = 1e-6;
    private static final double TOL = 1e-7;

    @Test
    void mulBackwardAMatchesFiniteDifferences() {
        double[] a = { 0.5, -0.3, 0.7, 0.1 };
        double[] b = { 0.2, 0.4, -0.1, 0.6 };
        double[] dC = { 0.5, -0.2, 0.3, 0.1 };

        double[] analytic = SplitQuat.mulBackwardA(b, dC);

        double maxErr = 0.0;
        for (int i = 0; i < 4; i++) {
            double orig = a[i];
            a[i] = orig + EPS;
            double lPlus = lossDot(SplitQuat.mul(a, b), dC);
            a[i] = orig - EPS;
            double lMinus = lossDot(SplitQuat.mul(a, b), dC);
            a[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - analytic[i]);
            if (err > maxErr) maxErr = err;
        }
        assertTrue(maxErr < TOL, "mulBackwardA error " + maxErr + " > " + TOL);
    }

    @Test
    void mulBackwardBMatchesFiniteDifferences() {
        double[] a = { 0.5, -0.3, 0.7, 0.1 };
        double[] b = { 0.2, 0.4, -0.1, 0.6 };
        double[] dC = { 0.5, -0.2, 0.3, 0.1 };

        double[] analytic = SplitQuat.mulBackwardB(a, dC);

        double maxErr = 0.0;
        for (int i = 0; i < 4; i++) {
            double orig = b[i];
            b[i] = orig + EPS;
            double lPlus = lossDot(SplitQuat.mul(a, b), dC);
            b[i] = orig - EPS;
            double lMinus = lossDot(SplitQuat.mul(a, b), dC);
            b[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - analytic[i]);
            if (err > maxErr) maxErr = err;
        }
        assertTrue(maxErr < TOL, "mulBackwardB error " + maxErr + " > " + TOL);
    }

    @Test
    void sandwichBackwardMatchesFiniteDifferences() {
        double[] q = { 0.6, -0.2, 0.5, 0.3 };
        double[] v = { 0.1, 0.4, -0.3, 0.2 };
        double[] dY = { 0.5, -0.2, 0.3, 0.1 };

        double[][] analytic = SplitQuat.sandwichBackward(q, v, dY);
        double[] dQ = analytic[0];
        double[] dV = analytic[1];

        double maxErr = 0.0;
        for (int i = 0; i < 4; i++) {
            double orig = q[i];
            q[i] = orig + EPS;
            double lPlus = lossDot(SplitQuat.sandwich(q, v), dY);
            q[i] = orig - EPS;
            double lMinus = lossDot(SplitQuat.sandwich(q, v), dY);
            q[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - dQ[i]);
            if (err > maxErr) maxErr = err;
        }
        for (int i = 0; i < 4; i++) {
            double orig = v[i];
            v[i] = orig + EPS;
            double lPlus = lossDot(SplitQuat.sandwich(q, v), dY);
            v[i] = orig - EPS;
            double lMinus = lossDot(SplitQuat.sandwich(q, v), dY);
            v[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - dV[i]);
            if (err > maxErr) maxErr = err;
        }
        assertTrue(maxErr < TOL, "sandwichBackward error " + maxErr + " > " + TOL);
    }

    @Test
    void normSignsIdempotentNullCone() {
        double[] ePlus = { 0.5, 0.0, 0.5, 0.0 };
        double[] eMinus = { 0.5, 0.0, -0.5, 0.0 };
        assertEquals(0.0, SplitQuat.norm(ePlus), 1e-15);
        assertEquals(0.0, SplitQuat.norm(eMinus), 1e-15);
    }

    private static double lossDot(double[] x, double[] g) {
        double s = 0;
        for (int i = 0; i < x.length; i++) s += x[i] * g[i];
        return s;
    }
}
