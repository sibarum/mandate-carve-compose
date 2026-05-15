package sibarum.strnn.demo;

import sibarum.strnn.hpb.HpbRegressionModel;
import sibarum.strnn.hpb.SmoothedBasisElement.Kernel;

/**
 * T7 from {@code docs/harmonic_piecewise_basis.md}: approximate a known
 * smooth periodic function with a linear readout over the harmonic
 * basis under three kernel choices (δ, box, tent). Sweep K; compare
 * MSE.
 *
 * <p>Uses closed-form least squares (normal equations) for the
 * readout, not SGD — T7 tests approximation <em>capacity</em>, not
 * training dynamics. Each (K, kernel) pair gets the optimal linear
 * readout in one step; reported MSE is the projection residual.
 *
 * <p>Target: {@code f(x) = exp(sin(2π x)) + sin(4π x) cos(6π x)}.
 * Smooth and periodic on [0, 1]; first term has all-positive Fourier
 * coefficients (Bessel-like decay); the product term simplifies via
 * product-to-sum to {@code (1/2)(sin(10π x) − sin(2π x))} and so
 * contributes content at fundamental frequencies 1 and 5.
 *
 * <p>Predicted ordering at fixed K (K ≥ 4, where the basis has
 * adequate rank): tent ≤ box ≤ δ in residual MSE. The smoothed bases
 * have higher continuity (C^0 for box, C^1 for tent), which matches
 * the smooth target better than piecewise-linear δ. F3 fires if no
 * advantage appears for smoothed kernels on this smooth target.
 */
public final class HpbSmoothApproximationDemo {

    private static final int N = 64;
    private static final int[] KS = { 2, 4, 8, 16 };
    private static final double[] WIDTH_FRACS = { 0.0625, 0.125, 0.25 };

    public static void main(String[] args) {
        double[] xs = new double[N];
        double[] ys = new double[N];
        for (int i = 0; i < N; i++) {
            xs[i] = (i + 0.5) / N;
            ys[i] = target(xs[i]);
        }
        double targetVar = variance(ys);

        System.out.println();
        System.out.println("=== HPB smooth approximation (T7) ===");
        System.out.println("  target: f(x) = exp(sin(2π x)) + sin(4π x) cos(6π x)");
        System.out.println("  N=" + N + " sample points, closed-form LS readout");
        System.out.printf("  target variance: %.6e (baseline for mean-only predictor)%n",
                targetVar);

        for (double wFrac : WIDTH_FRACS) {
            System.out.println();
            System.out.printf("--- w_k = %.4f · T_k ---%n", wFrac);
            System.out.printf("  %5s | %-14s | %-14s | %-14s | %-10s | %s%n",
                    "K", "delta MSE", "box MSE", "tent MSE", "tent/box", "best/delta");
            System.out.println("  ------+----------------+----------------+----------------+------------+-----------");

            for (int K : KS) {
                double mseDelta = leastSquaresMSE(K, Kernel.DELTA, wFrac, xs, ys);
                double mseBox   = leastSquaresMSE(K, Kernel.BOX,   wFrac, xs, ys);
                double mseTent  = leastSquaresMSE(K, Kernel.TENT,  wFrac, xs, ys);
                double r1 = mseBox > 0 ? mseTent / mseBox : 0.0;
                double best = Math.min(mseBox, mseTent);
                double r2 = mseDelta > 0 ? best / mseDelta : 0.0;
                System.out.printf("  K=%-3d | %.6e   | %.6e   | %.6e   | %.6f   | %.6f%n",
                        K, mseDelta, mseBox, mseTent, r1, r2);
            }
        }
    }

    private static double target(double x) {
        return Math.exp(Math.sin(2 * Math.PI * x))
                + Math.sin(4 * Math.PI * x) * Math.cos(6 * Math.PI * x);
    }

    private static double variance(double[] ys) {
        double mean = 0.0;
        for (double y : ys) mean += y;
        mean /= ys.length;
        double v = 0.0;
        for (double y : ys) v += (y - mean) * (y - mean);
        return v / ys.length;
    }

    private static double leastSquaresMSE(int K, Kernel kernel, double wFrac, double[] xs, double[] ys) {
        HpbRegressionModel m = new HpbRegressionModel(K, kernel, wFrac, 0L);
        int n = xs.length;
        int p = m.featDim() + 1;
        double[][] A = new double[n][p];
        for (int i = 0; i < n; i++) {
            double[] f = m.lift(xs[i]);
            System.arraycopy(f, 0, A[i], 0, m.featDim());
            A[i][p - 1] = 1.0;
        }
        double[] coefs = solveNormalEquations(A, ys);
        double mse = 0.0;
        for (int i = 0; i < n; i++) {
            double pred = 0.0;
            for (int j = 0; j < p; j++) pred += A[i][j] * coefs[j];
            double err = pred - ys[i];
            mse += err * err;
        }
        return mse / n;
    }

    private static double[] solveNormalEquations(double[][] A, double[] b) {
        int n = A.length;
        int p = A[0].length;
        double[][] AtA = new double[p][p];
        double[] Atb = new double[p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                Atb[j] += A[i][j] * b[i];
                for (int k = 0; k < p; k++) {
                    AtA[j][k] += A[i][j] * A[i][k];
                }
            }
        }
        return solveLinear(AtA, Atb);
    }

    private static double[] solveLinear(double[][] A, double[] b) {
        int n = A.length;
        double[][] M = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }
        for (int piv = 0; piv < n; piv++) {
            int maxRow = piv;
            for (int r = piv + 1; r < n; r++) {
                if (Math.abs(M[r][piv]) > Math.abs(M[maxRow][piv])) maxRow = r;
            }
            if (maxRow != piv) {
                double[] tmp = M[piv]; M[piv] = M[maxRow]; M[maxRow] = tmp;
            }
            if (Math.abs(M[piv][piv]) < 1e-14) {
                throw new ArithmeticException("singular system at pivot " + piv);
            }
            for (int r = piv + 1; r < n; r++) {
                double factor = M[r][piv] / M[piv][piv];
                for (int c = piv; c <= n; c++) M[r][c] -= factor * M[piv][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = M[i][n];
            for (int j = i + 1; j < n; j++) s -= M[i][j] * x[j];
            x[i] = s / M[i][i];
        }
        return x;
    }
}
