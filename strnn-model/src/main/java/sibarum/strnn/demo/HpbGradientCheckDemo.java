package sibarum.strnn.demo;

import sibarum.strnn.hpb.HpbModel;

/**
 * T3 from {@code docs/harmonic_piecewise_basis.md}: finite-difference
 * verification of {@link HpbModel#backward(int)}. Each learned scalar (W
 * and b) is perturbed by ±eps; central-difference (L(+eps) − L(−eps))/(2 eps)
 * is compared to the analytic gradient.
 *
 * <p>Test points are chosen mid-piece for k ∈ {1, 2} so that no perturbation
 * crosses a breakpoint (the basis is C^0 but has discontinuous derivative
 * in x at breakpoints; the parameter gradients are smooth in W and b
 * everywhere, but we keep x clear of breakpoints anyway to be safe).
 *
 * <p>Required before any optimization; falsifies F1 if it fires.
 */
public final class HpbGradientCheckDemo {

    private static final int K = 2;
    private static final int OUT_DIM = 2;
    private static final long SEED = 42L;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    public static void main(String[] args) {
        double[] testXs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        int target = 1;

        HpbModel model = new HpbModel(K, OUT_DIM, SEED);

        int totalChecked = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worstParam = "";

        for (double x : testXs) {
            model.zeroGrad();
            model.forward(x);
            model.backward(target);

            double[][] gw = model.gradWeights();
            for (int o = 0; o < OUT_DIM; o++) {
                for (int f = 0; f < model.featDim(); f++) {
                    double numeric = numericGradWeight(model, x, target, o, f);
                    String label = "W[" + o + "][" + f + "] @ x=" + x;
                    CheckResult r = check(label, numeric, gw[o][f]);
                    totalChecked++;
                    if (!r.pass) totalMismatch++;
                    if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
                    if (r.relErr > maxRelErr) maxRelErr = r.relErr;
                }
            }
            double[] gb = model.gradBiases();
            for (int o = 0; o < OUT_DIM; o++) {
                double numeric = numericGradBias(model, x, target, o);
                String label = "b[" + o + "] @ x=" + x;
                CheckResult r = check(label, numeric, gb[o]);
                totalChecked++;
                if (!r.pass) totalMismatch++;
                if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
                if (r.relErr > maxRelErr) maxRelErr = r.relErr;
            }
        }

        System.out.println();
        System.out.println("=== HPB gradient check (T3) ===");
        System.out.println("  K=" + K + ", outDim=" + OUT_DIM + ", featDim=" + (2 * K)
                + ", test points=" + testXs.length + ", eps=" + EPS);
        System.out.println("  checked: " + totalChecked + " parameters");
        System.out.println("  mismatches: " + totalMismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst: " + worstParam + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean passed = totalMismatch == 0;
        System.out.println("  RESULT: " + (passed ? "PASS" : "FAIL"));
        if (!passed) System.exit(1);
    }

    private static double numericGradWeight(HpbModel m, double x, int target, int o, int f) {
        double save = m.weights()[o][f];
        m.weights()[o][f] = save + EPS;
        m.forward(x);
        double lp = m.crossEntropyLoss(target);
        m.weights()[o][f] = save - EPS;
        m.forward(x);
        double lm = m.crossEntropyLoss(target);
        m.weights()[o][f] = save;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradBias(HpbModel m, double x, int target, int o) {
        double save = m.biases()[o];
        m.biases()[o] = save + EPS;
        m.forward(x);
        double lp = m.crossEntropyLoss(target);
        m.biases()[o] = save - EPS;
        m.forward(x);
        double lm = m.crossEntropyLoss(target);
        m.biases()[o] = save;
        return (lp - lm) / (2.0 * EPS);
    }

    private record CheckResult(boolean pass, double absErr, double relErr) {}

    private static CheckResult check(String label, double numeric, double analytic) {
        double absErr = Math.abs(numeric - analytic);
        double relErr = absErr / Math.max(Math.abs(numeric), 1e-12);
        boolean pass = absErr <= TOL_ABS || relErr <= TOL_REL;
        if (!pass) {
            System.out.printf("    MISMATCH %s: numeric=%.6e analytic=%.6e absErr=%.3e relErr=%.3e%n",
                    label, numeric, analytic, absErr, relErr);
        }
        return new CheckResult(pass, absErr, relErr);
    }
}
