package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Finite-difference gradient check for {@link Linear}. Per Trainable
 * contract, {@code backward(target)} computes the squared-error
 * gradient {@code (y − target)} and accumulates into pendingDw/pendingDb.
 *
 * <p>Loss: {@code L(W, b) = ½‖W·x + b − t‖²}. We finite-difference
 * each parameter and assert agreement with the analytic gradient at
 * ≤ 1e-9 max-abs-err.
 */
class LinearGradientCheckTest {

    private static final double EPS = 1e-5;
    private static final double TOL = 1e-9;

    @Test
    void linearWithBiasMatchesFiniteDifferences() {
        runCheck(true);
    }

    @Test
    void linearNoBiasMatchesFiniteDifferences() {
        runCheck(false);
    }

    private static void runCheck(boolean withBias) {
        int outDim = 3;
        int inDim = 4;
        Linear lin = new Linear(outDim, inDim, withBias, 42L);
        double[] x = { 0.7, -0.3, 1.1, -0.9 };
        double[] t = { 0.4, -0.6, 0.2 };

        MatrixValue mx = new MatrixValue(x);
        MatrixValue y0 = (MatrixValue) lin.apply(List.of(mx));
        // Gradient-flow shape: gradOut = y - t under MSE.
        double[] gradOut = new double[outDim];
        for (int i = 0; i < outDim; i++) gradOut[i] = y0.data()[i] - t[i];
        lin.backward(new MatrixValue(gradOut));

        // Analytic gradients per contract: dW[i,j] = dy[i] * x[j], db[i] = dy[i].
        double[][] analyticDw = new double[outDim][inDim];
        double[] analyticDb = withBias ? new double[outDim] : null;
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                analyticDw[i][j] = gradOut[i] * x[j];
            }
            if (withBias) analyticDb[i] = gradOut[i];
        }

        // Finite difference: for each parameter, perturb and recompute loss.
        double maxErr = 0.0;
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                double orig = lin.weights()[i][j];

                lin.weights()[i][j] = orig + EPS;
                MatrixValue yPlus = (MatrixValue) lin.apply(List.of(mx));
                double lPlus = halfMse(yPlus.data(), t);

                lin.weights()[i][j] = orig - EPS;
                MatrixValue yMinus = (MatrixValue) lin.apply(List.of(mx));
                double lMinus = halfMse(yMinus.data(), t);

                lin.weights()[i][j] = orig;
                double numeric = (lPlus - lMinus) / (2.0 * EPS);

                double err = Math.abs(numeric - analyticDw[i][j]);
                if (err > maxErr) maxErr = err;
            }
        }
        if (withBias) {
            for (int i = 0; i < outDim; i++) {
                double orig = lin.biases()[i];
                lin.biases()[i] = orig + EPS;
                MatrixValue yPlus = (MatrixValue) lin.apply(List.of(mx));
                double lPlus = halfMse(yPlus.data(), t);
                lin.biases()[i] = orig - EPS;
                MatrixValue yMinus = (MatrixValue) lin.apply(List.of(mx));
                double lMinus = halfMse(yMinus.data(), t);
                lin.biases()[i] = orig;
                double numeric = (lPlus - lMinus) / (2.0 * EPS);

                double err = Math.abs(numeric - analyticDb[i]);
                if (err > maxErr) maxErr = err;
            }
        }

        assertTrue(maxErr < TOL,
                "max abs gradient error " + maxErr + " exceeded tolerance " + TOL);
    }

    @Test
    void linearStepReducesLoss() {
        Linear lin = new Linear(2, 3, true, 7L);
        double[] x = { 0.1, -0.2, 0.5 };
        double[] t = { 1.0, -1.0 };
        MatrixValue mx = new MatrixValue(x);

        MatrixValue y0 = (MatrixValue) lin.apply(List.of(mx));
        double loss0 = halfMse(y0.data(), t);
        double[] gradOut = new double[2];
        for (int i = 0; i < 2; i++) gradOut[i] = y0.data()[i] - t[i];
        lin.backward(new MatrixValue(gradOut));
        lin.step(0.1);

        MatrixValue y1 = (MatrixValue) lin.apply(List.of(mx));
        double loss1 = halfMse(y1.data(), t);

        assertTrue(loss1 < loss0, "loss did not decrease after one SGD step");
    }

    private static double halfMse(double[] y, double[] t) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double d = y[i] - t[i];
            s += d * d;
        }
        return 0.5 * s;
    }
}
