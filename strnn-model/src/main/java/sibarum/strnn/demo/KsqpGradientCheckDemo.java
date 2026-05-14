package sibarum.strnn.demo;

import sibarum.strnn.ksqp.KsqpModel;

/**
 * Finite-difference gradient check for the rewritten KSQP. Verifies the
 * three gradient-trained parameter blocks — sq (per vocab), projection
 * matrices P_d (per active degree), head W and b — agree with their
 * analytic backwards at machine epsilon.
 *
 * <p>k (the per-vocab fixed random input) is frozen in training and
 * has no gradient slot in the model, so it isn't checked here. The
 * lift backward through k is exercised by {@code testLiftBackward}
 * separately if needed.
 *
 * <p>Three sub-runs exercise different active degrees:
 * <ul>
 *   <li>all tokens at p=1: only P_1 active</li>
 *   <li>all tokens at p=2: only P_2 active, cross-term monomials engaged</li>
 *   <li>mixed p across tokens: distinct P_d matrices touched at once</li>
 * </ul>
 */
public final class KsqpGradientCheckDemo {

    private static final int VOCAB = 3;
    private static final int OUT_DIM = 2;
    private static final int SEQ_LEN = 3;
    private static final int N = 4;
    private static final long SEED = 42L;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    public static void main(String[] args) {
        boolean ok1 = runCheck("all p=1 (linear lift)", new int[] {1, 1, 1});
        boolean ok2 = runCheck("all p=2 (quadratic lift, cross-terms)", new int[] {2, 2, 2});
        boolean ok3 = runCheck("mixed p (1, 2, 3 across tokens)", new int[] {1, 2, 3});
        boolean passed = ok1 && ok2 && ok3;
        System.out.println();
        System.out.println("OVERALL: " + (passed ? "PASS" : "FAIL"));
        if (!passed) System.exit(1);
    }

    private static boolean runCheck(String label, int[] pAssignment) {
        System.out.println();
        System.out.println("=== KSQP gradient check — " + label + " ===");

        int[] tokens = { 0, 1, 2 };
        int target = 1;

        KsqpModel model = new KsqpModel(VOCAB, OUT_DIM, SEQ_LEN, N, SEED);
        for (int v = 0; v < VOCAB; v++) model.setP(v, pAssignment[v]);

        model.zeroGrad();
        model.forward(tokens);
        model.backward(target);

        int totalChecked = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worstParam = "";

        // sq.
        double[][] sqGrad = model.sqGradients();
        for (int v = 0; v < VOCAB; v++) {
            for (int a = 0; a < KsqpModel.SQ_DIM; a++) {
                double numeric = numericGradSq(model, tokens, target, v, a);
                Stats s = check("sq[" + v + "][" + a + "]", numeric, sqGrad[v][a]);
                totalChecked++;
                if (!s.pass) totalMismatch++;
                if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "sq[" + v + "][" + a + "]"; }
                if (s.relErr > maxRelErr) maxRelErr = s.relErr;
            }
        }

        // Projection — only check the degrees actually used by the active token assignment.
        boolean[] activeDegree = new boolean[KsqpModel.P_MAX - KsqpModel.P_MIN + 1];
        for (int v = 0; v < VOCAB; v++) activeDegree[pAssignment[v] - KsqpModel.P_MIN] = true;
        for (int dIdx = 0; dIdx < activeDegree.length; dIdx++) {
            if (!activeDegree[dIdx]) continue;
            int d = KsqpModel.P_MIN + dIdx;
            double[][] gP = model.projectionGradFor(d);
            for (int r = 0; r < gP.length; r++) {
                for (int c = 0; c < KsqpModel.SQ_DIM; c++) {
                    double numeric = numericGradProjection(model, tokens, target, d, r, c);
                    Stats s = check("P_" + d + "[" + r + "][" + c + "]", numeric, gP[r][c]);
                    totalChecked++;
                    if (!s.pass) totalMismatch++;
                    if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "P_" + d + "[" + r + "][" + c + "]"; }
                    if (s.relErr > maxRelErr) maxRelErr = s.relErr;
                }
            }
        }

        // Head W, b.
        double[][] gw = model.headWGradients();
        for (int o = 0; o < OUT_DIM; o++) {
            for (int i = 0; i < gw[o].length; i++) {
                double numeric = numericGradHeadW(model, tokens, target, o, i);
                Stats s = check("W[" + o + "][" + i + "]", numeric, gw[o][i]);
                totalChecked++;
                if (!s.pass) totalMismatch++;
                if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "W[" + o + "][" + i + "]"; }
                if (s.relErr > maxRelErr) maxRelErr = s.relErr;
            }
        }
        double[] gb = model.headBGradients();
        for (int o = 0; o < OUT_DIM; o++) {
            double numeric = numericGradHeadB(model, tokens, target, o);
            Stats s = check("b[" + o + "]", numeric, gb[o]);
            totalChecked++;
            if (!s.pass) totalMismatch++;
            if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "b[" + o + "]"; }
            if (s.relErr > maxRelErr) maxRelErr = s.relErr;
        }

        System.out.println("  checked: " + totalChecked + " parameters");
        System.out.println("  mismatches: " + totalMismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst param: " + worstParam + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean passed = totalMismatch == 0;
        System.out.println("  RESULT: " + (passed ? "PASS" : "FAIL"));
        return passed;
    }

    private static double numericGradSq(KsqpModel model, int[] tokens, int target, int v, int a) {
        double[] sq = model.sq(v);
        double orig = sq[a];
        sq[a] = orig + EPS;
        model.forward(tokens);
        double lp = model.crossEntropyLoss(target);
        sq[a] = orig - EPS;
        model.forward(tokens);
        double lm = model.crossEntropyLoss(target);
        sq[a] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradProjection(KsqpModel model, int[] tokens, int target,
                                                int d, int r, int c) {
        double[][] P = model.projectionFor(d);
        double orig = P[r][c];
        P[r][c] = orig + EPS;
        model.forward(tokens);
        double lp = model.crossEntropyLoss(target);
        P[r][c] = orig - EPS;
        model.forward(tokens);
        double lm = model.crossEntropyLoss(target);
        P[r][c] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradHeadW(KsqpModel model, int[] tokens, int target, int o, int i) {
        double[][] w = model.headWeights();
        double orig = w[o][i];
        w[o][i] = orig + EPS;
        model.forward(tokens);
        double lp = model.crossEntropyLoss(target);
        w[o][i] = orig - EPS;
        model.forward(tokens);
        double lm = model.crossEntropyLoss(target);
        w[o][i] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradHeadB(KsqpModel model, int[] tokens, int target, int o) {
        double[] b = model.headBiases();
        double orig = b[o];
        b[o] = orig + EPS;
        model.forward(tokens);
        double lp = model.crossEntropyLoss(target);
        b[o] = orig - EPS;
        model.forward(tokens);
        double lm = model.crossEntropyLoss(target);
        b[o] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static Stats check(String label, double numeric, double analytic) {
        double absErr = Math.abs(numeric - analytic);
        double denom = Math.max(1.0, Math.max(Math.abs(numeric), Math.abs(analytic)));
        double relErr = absErr / denom;
        boolean pass = (absErr < TOL_ABS) || (relErr < TOL_REL);
        if (!pass) {
            System.out.println("  MISMATCH " + label
                    + " analytic=" + String.format("%+.6e", analytic)
                    + " numeric=" + String.format("%+.6e", numeric)
                    + " absErr=" + String.format("%.3e", absErr)
                    + " relErr=" + String.format("%.3e", relErr));
        }
        return new Stats(pass, absErr, relErr);
    }

    private record Stats(boolean pass, double absErr, double relErr) {}
}
