package sibarum.strnn.demo;

import sibarum.strnn.ksqp.KsqpKvModel;

/**
 * Finite-difference gradient check for the soft-attention KV model.
 * Verifies sq, P_d (per active degree), head W and b. storedKeys are
 * frozen and have no gradient slot; query is input data and likewise
 * isn't a model parameter. The softmax-attention pathway flows into sq
 * and P_d through dYper = w_v · dYAgg, so the check at p > 1 exercises
 * the whole attention-and-sandwich chain.
 */
public final class KsqpKvGradientCheckDemo {

    private static final int OUT_DIM = 2;
    private static final int QUERY_DIM = 2;
    private static final long SEED = 42L;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    private static final double[][] STORED_KEYS = {
            { 1.0,  1.0 },
            { 1.0, -1.0 },
            {-1.0, -1.0 },
            {-1.0,  1.0 }
    };
    private static final double[] QUERY = { 0.7, 0.3 }; // off-center, not on any prototype

    public static void main(String[] args) {
        boolean ok1 = runCheck("frozen keys, all p=1 (linear lift)", new int[] {1, 1, 1, 1}, false);
        boolean ok2 = runCheck("frozen keys, all p=2 (quadratic lift, cross-terms)", new int[] {2, 2, 2, 2}, false);
        boolean ok3 = runCheck("frozen keys, mixed p (1, 2, 3, 4)", new int[] {1, 2, 3, 4}, false);
        boolean ok4 = runCheck("trainable keys, all p=1", new int[] {1, 1, 1, 1}, true);
        boolean ok5 = runCheck("trainable keys, all p=2", new int[] {2, 2, 2, 2}, true);
        boolean ok6 = runCheck("trainable keys, mixed p (1, 2, 3, 4)", new int[] {1, 2, 3, 4}, true);
        boolean passed = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
        System.out.println();
        System.out.println("OVERALL: " + (passed ? "PASS" : "FAIL"));
        if (!passed) System.exit(1);
    }

    private static boolean runCheck(String label, int[] pAssignment, boolean trainStoredKeys) {
        System.out.println();
        System.out.println("=== KSQP-KV gradient check — " + label + " ===");

        int target = 1;

        KsqpKvModel model = trainStoredKeys
                ? new KsqpKvModel(OUT_DIM, QUERY_DIM, /*M=*/STORED_KEYS.length, SEED,
                        KsqpKvModel.DEFAULT_SQ_INIT_A0, KsqpKvModel.DEFAULT_SQ_INIT_NOISE,
                        KsqpKvModel.DEFAULT_TEMPERATURE,
                        KsqpKvModel.DEFAULT_STORED_KEY_INIT_RANGE)
                : new KsqpKvModel(OUT_DIM, QUERY_DIM, STORED_KEYS, SEED);
        for (int v = 0; v < pAssignment.length; v++) model.setP(v, pAssignment[v]);

        model.zeroGrad();
        model.forward(QUERY);
        model.backward(target);

        int totalChecked = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worstParam = "";

        double[][] sqGrad = model.sqGradients();
        for (int v = 0; v < pAssignment.length; v++) {
            for (int a = 0; a < KsqpKvModel.SQ_DIM; a++) {
                double numeric = numericGradSq(model, target, v, a);
                Stats s = check("sq[" + v + "][" + a + "]", numeric, sqGrad[v][a]);
                totalChecked++;
                if (!s.pass) totalMismatch++;
                if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "sq[" + v + "][" + a + "]"; }
                if (s.relErr > maxRelErr) maxRelErr = s.relErr;
            }
        }

        boolean[] activeDegree = new boolean[KsqpKvModel.P_MAX - KsqpKvModel.P_MIN + 1];
        for (int d : pAssignment) activeDegree[d - KsqpKvModel.P_MIN] = true;
        for (int dIdx = 0; dIdx < activeDegree.length; dIdx++) {
            if (!activeDegree[dIdx]) continue;
            int d = KsqpKvModel.P_MIN + dIdx;
            double[][] gP = model.projectionGradFor(d);
            for (int r = 0; r < gP.length; r++) {
                for (int c = 0; c < KsqpKvModel.SQ_DIM; c++) {
                    double numeric = numericGradProjection(model, target, d, r, c);
                    Stats s = check("P_" + d + "[" + r + "][" + c + "]", numeric, gP[r][c]);
                    totalChecked++;
                    if (!s.pass) totalMismatch++;
                    if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "P_" + d + "[" + r + "][" + c + "]"; }
                    if (s.relErr > maxRelErr) maxRelErr = s.relErr;
                }
            }
        }

        double[][] gw = model.headWGradients();
        for (int o = 0; o < OUT_DIM; o++) {
            for (int i = 0; i < gw[o].length; i++) {
                double numeric = numericGradHeadW(model, target, o, i);
                Stats s = check("W[" + o + "][" + i + "]", numeric, gw[o][i]);
                totalChecked++;
                if (!s.pass) totalMismatch++;
                if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "W[" + o + "][" + i + "]"; }
                if (s.relErr > maxRelErr) maxRelErr = s.relErr;
            }
        }
        double[] gb = model.headBGradients();
        for (int o = 0; o < OUT_DIM; o++) {
            double numeric = numericGradHeadB(model, target, o);
            Stats s = check("b[" + o + "]", numeric, gb[o]);
            totalChecked++;
            if (!s.pass) totalMismatch++;
            if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "b[" + o + "]"; }
            if (s.relErr > maxRelErr) maxRelErr = s.relErr;
        }

        if (trainStoredKeys) {
            double[][] gK = model.storedKeysGradients();
            for (int v = 0; v < gK.length; v++) {
                for (int j = 0; j < QUERY_DIM; j++) {
                    double numeric = numericGradStoredKey(model, target, v, j);
                    Stats s = check("k[" + v + "][" + j + "]", numeric, gK[v][j]);
                    totalChecked++;
                    if (!s.pass) totalMismatch++;
                    if (s.absErr > maxAbsErr) { maxAbsErr = s.absErr; worstParam = "k[" + v + "][" + j + "]"; }
                    if (s.relErr > maxRelErr) maxRelErr = s.relErr;
                }
            }
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

    private static double numericGradSq(KsqpKvModel model, int target, int v, int a) {
        double[] sq = model.sq(v);
        double orig = sq[a];
        sq[a] = orig + EPS;
        model.forward(QUERY);
        double lp = model.crossEntropyLoss(target);
        sq[a] = orig - EPS;
        model.forward(QUERY);
        double lm = model.crossEntropyLoss(target);
        sq[a] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradProjection(KsqpKvModel model, int target, int d, int r, int c) {
        double[][] P = model.projectionFor(d);
        double orig = P[r][c];
        P[r][c] = orig + EPS;
        model.forward(QUERY);
        double lp = model.crossEntropyLoss(target);
        P[r][c] = orig - EPS;
        model.forward(QUERY);
        double lm = model.crossEntropyLoss(target);
        P[r][c] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradHeadW(KsqpKvModel model, int target, int o, int i) {
        double[][] w = model.headWeights();
        double orig = w[o][i];
        w[o][i] = orig + EPS;
        model.forward(QUERY);
        double lp = model.crossEntropyLoss(target);
        w[o][i] = orig - EPS;
        model.forward(QUERY);
        double lm = model.crossEntropyLoss(target);
        w[o][i] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradStoredKey(KsqpKvModel model, int target, int v, int j) {
        double[] k = model.storedKey(v);
        double orig = k[j];
        k[j] = orig + EPS;
        model.forward(QUERY);
        double lp = model.crossEntropyLoss(target);
        k[j] = orig - EPS;
        model.forward(QUERY);
        double lm = model.crossEntropyLoss(target);
        k[j] = orig;
        return (lp - lm) / (2.0 * EPS);
    }

    private static double numericGradHeadB(KsqpKvModel model, int target, int o) {
        double[] b = model.headBiases();
        double orig = b[o];
        b[o] = orig + EPS;
        model.forward(QUERY);
        double lp = model.crossEntropyLoss(target);
        b[o] = orig - EPS;
        model.forward(QUERY);
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
