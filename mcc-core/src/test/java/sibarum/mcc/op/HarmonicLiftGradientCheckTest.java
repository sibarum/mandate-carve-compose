package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.op.advanced.SmoothedBasisElement.Kernel;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Finite-difference gradient checks for {@link HarmonicLift}'s backward.
 * Loss: {@code L = ⟨gradOut, lift(x)⟩}, so the analytic backward result
 * {@code dL/dx[i]} should match the numeric central-difference value.
 *
 * <p>Test points are chosen mid-piece for the relevant basis frequencies
 * so finite-difference perturbations never cross a breakpoint (the lift
 * is C^0 under δ kernel, smooth under box/tent, so dL/dx is well-defined
 * everywhere except at the original-basis breakpoints).
 */
class HarmonicLiftGradientCheckTest {

    private static final double EPS = 1e-6;
    private static final double TOL_DELTA = 1e-8;
    private static final double TOL_SMOOTHED = 1e-7;

    @Test
    void deltaKernelGradientMatchesFiniteDifferences() {
        runCheck(Kernel.DELTA, 0.0, TOL_DELTA);
    }

    @Test
    void boxKernelGradientMatchesFiniteDifferences() {
        runCheck(Kernel.BOX, 0.125, TOL_SMOOTHED);
    }

    @Test
    void tentKernelGradientMatchesFiniteDifferences() {
        runCheck(Kernel.TENT, 0.125, TOL_SMOOTHED);
    }

    private static void runCheck(Kernel kernel, double widthFrac, double tol) {
        int K = 2;
        int inputDim = 3;
        HarmonicLift lift = new HarmonicLift(K, inputDim, kernel, widthFrac);
        // Test points chosen to avoid kinks: away from sq_k's original
        // breakpoints (DELTA case) AND away from breakpoint ± w/2 positions
        // where the smoothed sq_k has C^0 kinks (BOX case). For DELTA,
        // sq_k's breakpoints are at i/(4k); for BOX with w_k = T_k/8,
        // smoothed kinks are at sq_k.breakpoint ± 1/(16k). The points
        // {0.1, 0.4, 0.7} clear all of these for k ≤ 2.
        double[] x = { 0.1, 0.4, 0.7 };
        MatrixValue mx = new MatrixValue(x);
        MatrixValue out = (MatrixValue) lift.apply(List.of(mx));

        // Choose a non-trivial gradOut so dL/dx[i] depends on multiple features.
        double[] gradOut = new double[out.data().length];
        for (int i = 0; i < gradOut.length; i++) {
            gradOut[i] = 0.5 - (i % 5) * 0.1;
        }
        List<Value> dx = lift.backward(new MatrixValue(gradOut));
        double[] analytic = ((MatrixValue) dx.getFirst()).data();
        assertEquals(inputDim, analytic.length);

        double maxErr = 0.0;
        int worstIdx = -1;
        for (int i = 0; i < inputDim; i++) {
            double orig = x[i];
            x[i] = orig + EPS;
            MatrixValue yPlus = (MatrixValue) lift.apply(List.of(new MatrixValue(x)));
            double lPlus = dot(gradOut, yPlus.data());

            x[i] = orig - EPS;
            MatrixValue yMinus = (MatrixValue) lift.apply(List.of(new MatrixValue(x)));
            double lMinus = dot(gradOut, yMinus.data());

            x[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * EPS);
            double err = Math.abs(numeric - analytic[i]);
            if (err > maxErr) {
                maxErr = err;
                worstIdx = i;
            }
        }
        assertTrue(maxErr < tol,
                "kernel=" + kernel + " max abs gradient error " + maxErr
                        + " exceeded tolerance " + tol + " at index " + worstIdx);
    }

    @Test
    void liftDimensionsMatchSpec() {
        HarmonicLift lift = new HarmonicLift(4, 3, Kernel.DELTA, 0.0);
        assertEquals(3, lift.inputDim());
        assertEquals(4, lift.K());
        assertEquals(3 * 8, lift.outDim());
        MatrixValue out = (MatrixValue) lift.apply(List.of(
                new MatrixValue(new double[] { 0.1, 0.2, 0.3 })));
        assertEquals(24, out.data().length);
    }

    @Test
    void deltaKernelLiftMatchesRawBasis() {
        // At x = 1/16, the K=1 lift should produce [tri_1(1/16), sq_1(1/16)] =
        // [0.25, +4] (peak slope of piece 0).
        HarmonicLift lift = new HarmonicLift(1, 1, Kernel.DELTA, 0.0);
        MatrixValue out = (MatrixValue) lift.apply(List.of(
                new MatrixValue(new double[] { 1.0 / 16 })));
        assertEquals(2, out.data().length);
        assertEquals(0.25, out.data()[0], 1e-12);
        assertEquals(4.0, out.data()[1], 1e-12);
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
