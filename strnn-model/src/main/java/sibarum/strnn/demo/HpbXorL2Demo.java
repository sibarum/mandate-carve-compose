package sibarum.strnn.demo;

import sibarum.strnn.hpb.HarmonicBasis;
import sibarum.strnn.hpb.PiecewisePolynomial;

import java.util.Random;

/**
 * L2-against-labels variant of 1D K=1 XOR. Unlike CE, L2 has a
 * finite-norm minimum: the 4-equation 3-unknown overdetermined LS
 * system has a unique closed-form optimum at
 * {@code (w_tri, w_sq, b) = (0, -1/8, 1/2)} achieving exactly zero
 * residual (the labels happen to lie in the column-span of the
 * feature+bias matrix).
 *
 * <p>This is the right test for the doc's §5 F2 "exact-rational
 * basin" claim. Under L2, "exact basin" is meaningful: it names the
 * specific finite (W*, b*) where the residual is zero. If SGD on L2
 * converges to that point cleanly, the claim is supported; if it
 * settles elsewhere, the claim is wrong.
 *
 * <p>Half-MSE loss: {@code L = (1/(2n)) Σ (pred_i - y_i)²}.
 * Single linear output {@code pred = w_tri·tri_1(x) + w_sq·sq_1(x) + b}.
 *
 * <p>Predicted: 5/5 seeds converge to ‖W − W*‖ ≤ 1e-10 within ~50k
 * epochs.
 */
public final class HpbXorL2Demo {

    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L };
    private static final int[] CHECKPOINTS = { 1000, 5000, 20000, 100000 };
    private static final double LR = 0.05;

    public static void main(String[] args) {
        double[] xs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        double[] ys = { 0.0, 1.0, 1.0, 0.0 };

        double W_TRI_STAR = 0.0;
        double W_SQ_STAR  = -0.125;
        double B_STAR     = 0.5;

        PiecewisePolynomial tri = HarmonicBasis.triK(1);
        PiecewisePolynomial sq  = HarmonicBasis.sqK(1);

        System.out.println();
        System.out.println("=== HPB 1D K=1 XOR — L2 against labels {0,1} ===");
        System.out.printf("  predicted closed-form optimum: w_tri=%.4f, w_sq=%.4f, b=%.4f%n",
                W_TRI_STAR, W_SQ_STAR, B_STAR);
        System.out.printf("  4 features at x ∈ {1/16, 5/16, 9/16, 13/16}; labels in column-span%n");
        System.out.printf("  lr=%.4f, %d seeds%n", LR, SEEDS.length);
        System.out.println();

        double mseAtOpt = 0.0;
        for (int i = 0; i < xs.length; i++) {
            double f0 = tri.evaluate(xs[i]);
            double f1 = sq.evaluate(xs[i]);
            double pred = W_TRI_STAR * f0 + W_SQ_STAR * f1 + B_STAR;
            double err = pred - ys[i];
            mseAtOpt += err * err;
        }
        mseAtOpt /= (2.0 * xs.length);
        System.out.printf("  sanity: half-MSE at predicted optimum = %.3e (should be 0)%n",
                mseAtOpt);
        System.out.println();

        for (long seed : SEEDS) {
            System.out.printf("--- seed %d ---%n", seed);
            System.out.printf("  %8s | %-12s | %-15s | %-15s | %-15s | %s%n",
                    "epochs", "half-MSE", "w_tri", "w_sq", "b", "‖W - W*‖");

            Random rng = new Random(seed);
            double init = 0.5;
            double w0 = (rng.nextDouble() * 2.0 - 1.0) * init;
            double w1 = (rng.nextDouble() * 2.0 - 1.0) * init;
            double b  = (rng.nextDouble() * 2.0 - 1.0) * init;

            int prev = 0;
            for (int target : CHECKPOINTS) {
                int delta = target - prev;
                for (int e = 0; e < delta; e++) {
                    double gw0 = 0.0, gw1 = 0.0, gb = 0.0;
                    for (int i = 0; i < xs.length; i++) {
                        double f0 = tri.evaluate(xs[i]);
                        double f1 = sq.evaluate(xs[i]);
                        double pred = w0 * f0 + w1 * f1 + b;
                        double err = pred - ys[i];
                        gw0 += err * f0;
                        gw1 += err * f1;
                        gb  += err;
                    }
                    double scale = LR / xs.length;
                    w0 -= scale * gw0;
                    w1 -= scale * gw1;
                    b  -= scale * gb;
                }
                prev = target;

                double mse = 0.0;
                for (int i = 0; i < xs.length; i++) {
                    double f0 = tri.evaluate(xs[i]);
                    double f1 = sq.evaluate(xs[i]);
                    double pred = w0 * f0 + w1 * f1 + b;
                    double err = pred - ys[i];
                    mse += err * err;
                }
                mse /= (2.0 * xs.length);

                double dx = w0 - W_TRI_STAR;
                double dy = w1 - W_SQ_STAR;
                double dz = b  - B_STAR;
                double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                System.out.printf("  %8d | %.4e   | %+.8e | %+.8e | %+.8e | %.4e%n",
                        target, mse, w0, w1, b, dist);
            }
            System.out.println();
        }
    }
}
