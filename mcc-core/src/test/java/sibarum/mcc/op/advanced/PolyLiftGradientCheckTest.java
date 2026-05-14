package sibarum.mcc.op.advanced;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Finite-difference gradient check for {@link PolyLift#liftBackward}.
 * Loss is {@code L = ⟨dM, m⟩}, so the analytic backward result
 * {@code liftBackward(k, monomials, dM)} should match
 * {@code dL/dk_j = dM · ∂m/∂k_j} elementwise.
 */
class PolyLiftGradientCheckTest {

    private static final double EPS = 1e-6;
    private static final double TOL = 1e-7;

    @Test
    void monomialCountMatchesBinomial() {
        assertEquals(1, PolyLift.monomialCount(4, 0));
        assertEquals(4, PolyLift.monomialCount(4, 1));
        assertEquals(10, PolyLift.monomialCount(4, 2));
        assertEquals(20, PolyLift.monomialCount(4, 3));
        assertEquals(35, PolyLift.monomialCount(4, 4));
    }

    @Test
    void enumerationProducesCorrectCount() {
        int[][] ms = PolyLift.enumerateMonomials(4, 2);
        assertEquals(10, ms.length);
        for (int[] alpha : ms) {
            int sum = 0;
            for (int a : alpha) sum += a;
            assertEquals(2, sum, "Each multi-index must sum to d=2");
        }
    }

    @Test
    void liftBackwardDegree3MatchesFiniteDifferences() {
        int n = 3;
        int d = 3;
        int[][] monomials = PolyLift.enumerateMonomials(n, d);
        double[] k = { 0.7, -0.5, 1.2 };
        double[] dM = new double[monomials.length];
        for (int i = 0; i < dM.length; i++) dM[i] = (i * 0.13) - 0.4;

        double[] analytic = PolyLift.liftBackward(k, monomials, dM);

        double maxErr = 0.0;
        for (int j = 0; j < n; j++) {
            double orig = k[j];
            k[j] = orig + EPS;
            double lPlus = lossDot(PolyLift.lift(k, monomials), dM);
            k[j] = orig - EPS;
            double lMinus = lossDot(PolyLift.lift(k, monomials), dM);
            k[j] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - analytic[j]);
            if (err > maxErr) maxErr = err;
        }
        assertTrue(maxErr < TOL, "PolyLift backward error " + maxErr + " > " + TOL);
    }

    private static double lossDot(double[] x, double[] g) {
        double s = 0;
        for (int i = 0; i < x.length; i++) s += x[i] * g[i];
        return s;
    }
}
