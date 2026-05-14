package sibarum.mcc.op.block;

import org.junit.jupiter.api.Test;
import sibarum.mcc.value.MatrixValue;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end finite-difference gradient check on {@link MlpBlock}.
 * Compares the analytic gradients (held inside {@link Mlp}'s
 * {@code gradWeights}/{@code gradBiases}) against numerical
 * derivatives of {@code ½‖forward(x) − target‖²} with respect to each
 * weight and bias.
 *
 * <p>Verifies the MLP's backprop through arbitrary layer counts —
 * 3 hidden layers with mixed widths exercises the chain rule through
 * multiple ReLU thresholds and the linear output.
 */
class MlpBlockGradientCheckTest {

    private static final double EPS = 1e-5;
    private static final double TOL = 1e-7;

    @Test
    void mlpGradientsMatchFiniteDifferences() {
        int[] sizes = { 3, 5, 4, 2 };
        long seed = 13L;
        Mlp mlp = new Mlp(sizes, seed);
        MlpBlock block = new MlpBlock(mlp);
        double[] x = { 0.5, -0.7, 0.3 };
        double[] t = { 0.2, -0.4 };

        // Run forward + backward to populate gradients.
        MatrixValue y0 = (MatrixValue) block.apply(List.of(new MatrixValue(x)));
        block.backward(new MatrixValue(t));

        // Snapshot analytic gradients.
        double[][][] analyticDw = deepCopy(mlp.weights(), accessGradWeights(mlp));
        double[][] analyticDb = deepCopy(mlp.biases(), accessGradBiases(mlp));

        double maxErr = 0.0;
        for (int l = 0; l < mlp.weights().length; l++) {
            double[][] w = mlp.weights()[l];
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[i].length; j++) {
                    double orig = w[i][j];
                    w[i][j] = orig + EPS;
                    double lPlus = halfMse(mlp.forward(x), t);
                    w[i][j] = orig - EPS;
                    double lMinus = halfMse(mlp.forward(x), t);
                    w[i][j] = orig;
                    double numeric = (lPlus - lMinus) / (2.0 * EPS);
                    double err = Math.abs(numeric - analyticDw[l][i][j]);
                    if (err > maxErr) maxErr = err;
                }
            }
        }
        for (int l = 0; l < mlp.biases().length; l++) {
            double[] b = mlp.biases()[l];
            for (int i = 0; i < b.length; i++) {
                double orig = b[i];
                b[i] = orig + EPS;
                double lPlus = halfMse(mlp.forward(x), t);
                b[i] = orig - EPS;
                double lMinus = halfMse(mlp.forward(x), t);
                b[i] = orig;
                double numeric = (lPlus - lMinus) / (2.0 * EPS);
                double err = Math.abs(numeric - analyticDb[l][i]);
                if (err > maxErr) maxErr = err;
            }
        }
        assertTrue(maxErr < TOL,
                "MlpBlock max gradient error " + maxErr + " > " + TOL);
        assertNotNull(y0);
    }

    @Test
    void mlpStepReducesLoss() {
        Mlp mlp = new Mlp(new int[] { 4, 6, 2 }, 99L);
        MlpBlock block = new MlpBlock(mlp);
        double[] x = { 0.1, -0.2, 0.3, -0.4 };
        double[] t = { 1.0, 0.5 };

        MatrixValue y0 = (MatrixValue) block.apply(List.of(new MatrixValue(x)));
        double loss0 = halfMse(y0.data(), t);
        block.backward(new MatrixValue(t));
        block.step(0.05);

        MatrixValue y1 = (MatrixValue) block.apply(List.of(new MatrixValue(x)));
        double loss1 = halfMse(y1.data(), t);
        assertTrue(loss1 < loss0,
                "loss did not decrease after one step: " + loss0 + " -> " + loss1);
    }

    private static double halfMse(double[] y, double[] t) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double d = y[i] - t[i];
            s += d * d;
        }
        return 0.5 * s;
    }

    private static double[][][] deepCopy(double[][][] template, double[][][] src) {
        double[][][] out = new double[template.length][][];
        for (int l = 0; l < template.length; l++) {
            out[l] = new double[template[l].length][];
            for (int i = 0; i < template[l].length; i++) {
                out[l][i] = src[l][i].clone();
            }
        }
        return out;
    }

    private static double[][] deepCopy(double[][] template, double[][] src) {
        double[][] out = new double[template.length][];
        for (int l = 0; l < template.length; l++) {
            out[l] = src[l].clone();
        }
        return out;
    }

    // Helpers to pull the gradient arrays out of the Mlp.
    // Mlp's gradWeights/gradBiases are private; we access them by reading after
    // backward. Since Mlp doesn't expose them, we use reflection to snapshot.
    private static double[][][] accessGradWeights(Mlp mlp) {
        try {
            java.lang.reflect.Field f = Mlp.class.getDeclaredField("gradWeights");
            f.setAccessible(true);
            return (double[][][]) f.get(mlp);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] accessGradBiases(Mlp mlp) {
        try {
            java.lang.reflect.Field f = Mlp.class.getDeclaredField("gradBiases");
            f.setAccessible(true);
            return (double[][]) f.get(mlp);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
}
