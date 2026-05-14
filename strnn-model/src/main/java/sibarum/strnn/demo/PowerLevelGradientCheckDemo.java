package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.ElevatorEmbeddingTable;
import sibarum.strnn.ksq.elevator.ElevatorOutputHead;
import sibarum.strnn.ksq.elevator.PowerLevelModel;

/**
 * Finite-difference gradient check for iter 7's PowerLevelModel. The new
 * backward pathway is the signed-power activation: each component
 * y_i = sign(s_i) · |s_i|^n with chain rules through both ∂y/∂s and
 * ∂y/∂n. The latter is the new scalar gradient on the level parameter n.
 *
 * <p>Run at n = 1.5 (a non-degenerate value: not the identity case n=1
 * where some chain factors collapse, and not the singularity-prone n<1
 * regime). This ensures we exercise the actual nonlinear backward, not
 * a special case.
 */
public final class PowerLevelGradientCheckDemo {

    private static final int VOCAB = 4;
    private static final int OUT_DIM = 3;
    private static final long SEED = 42L;
    private static final double N_INIT = 1.5;
    private static final double EMBED_INIT = 1.0;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    public static void main(String[] args) {
        int[] tokens = { 0, 1, 2 };
        int target = 1;

        PowerLevelModel model = new PowerLevelModel(VOCAB, OUT_DIM, SEED, EMBED_INIT, N_INIT);

        model.zeroGrad();
        model.forward(tokens);
        model.backward(target);

        int totalChecked = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worstParam = "";

        ElevatorEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < ElevatorAnchors.N; a++) {
                double numeric = numericGradEmbedding(model, tokens, target, v, a);
                String label = "E[" + v + "][" + a + "]";
                CheckResult r = check(label, numeric, analyticGrad[a]);
                totalChecked++;
                if (!r.pass) totalMismatch++;
                if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
                if (r.relErr > maxRelErr) maxRelErr = r.relErr;
            }
        }

        ElevatorOutputHead head = model.head();
        double[][] gw = head.gradWeights();
        for (int k = 0; k < OUT_DIM; k++) {
            for (int i = 0; i < head.inputDim(); i++) {
                double numeric = numericGradHeadWeight(model, tokens, target, k, i);
                String label = "W[" + k + "][" + i + "]";
                CheckResult r = check(label, numeric, gw[k][i]);
                totalChecked++;
                if (!r.pass) totalMismatch++;
                if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
                if (r.relErr > maxRelErr) maxRelErr = r.relErr;
            }
        }
        double[] gb = head.gradBiases();
        for (int k = 0; k < OUT_DIM; k++) {
            double numeric = numericGradHeadBias(model, tokens, target, k);
            String label = "b[" + k + "]";
            CheckResult r = check(label, numeric, gb[k]);
            totalChecked++;
            if (!r.pass) totalMismatch++;
            if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
            if (r.relErr > maxRelErr) maxRelErr = r.relErr;
        }

        // The new scalar gradient: ∂L/∂n.
        double numericGradN = numericGradN(model, tokens, target);
        double analyticGradN = model.gradN();
        CheckResult rn = check("n", numericGradN, analyticGradN);
        totalChecked++;
        if (!rn.pass) totalMismatch++;
        if (rn.absErr > maxAbsErr) { maxAbsErr = rn.absErr; worstParam = "n"; }
        if (rn.relErr > maxRelErr) maxRelErr = rn.relErr;

        System.out.println();
        System.out.println("=== PowerLevelModel gradient check (signed-power activation) ===");
        System.out.println("  T=" + tokens.length + ", vocab=" + VOCAB
                + ", n_anchors=" + ElevatorAnchors.N + ", outDim=" + OUT_DIM
                + ", n_init=" + N_INIT + ", eps=" + EPS);
        System.out.println("  analytic ∂L/∂n = " + String.format("%+.6e", analyticGradN)
                + "    numeric = " + String.format("%+.6e", numericGradN));
        System.out.println("  checked: " + totalChecked + " parameters (incl. scalar n)");
        System.out.println("  mismatches: " + totalMismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst param: " + worstParam + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean passed = totalMismatch == 0;
        System.out.println("  RESULT: " + (passed ? "PASS" : "FAIL"));
        if (!passed) System.exit(1);
    }

    private static double numericGradEmbedding(PowerLevelModel model, int[] tokens, int target,
                                                int tokenId, int anchorIdx) {
        double[] row = model.embedding().lookup(tokenId);
        double orig = row[anchorIdx];
        row[anchorIdx] = orig + EPS;
        model.forward(tokens);
        double lPlus = model.crossEntropyLoss(target);
        row[anchorIdx] = orig - EPS;
        model.forward(tokens);
        double lMinus = model.crossEntropyLoss(target);
        row[anchorIdx] = orig;
        return (lPlus - lMinus) / (2.0 * EPS);
    }

    private static double numericGradHeadWeight(PowerLevelModel model, int[] tokens, int target,
                                                 int outIdx, int inIdx) {
        double[][] w = model.head().weights();
        double orig = w[outIdx][inIdx];
        w[outIdx][inIdx] = orig + EPS;
        model.forward(tokens);
        double lPlus = model.crossEntropyLoss(target);
        w[outIdx][inIdx] = orig - EPS;
        model.forward(tokens);
        double lMinus = model.crossEntropyLoss(target);
        w[outIdx][inIdx] = orig;
        return (lPlus - lMinus) / (2.0 * EPS);
    }

    private static double numericGradHeadBias(PowerLevelModel model, int[] tokens, int target, int outIdx) {
        double[] b = model.head().biases();
        double orig = b[outIdx];
        b[outIdx] = orig + EPS;
        model.forward(tokens);
        double lPlus = model.crossEntropyLoss(target);
        b[outIdx] = orig - EPS;
        model.forward(tokens);
        double lMinus = model.crossEntropyLoss(target);
        b[outIdx] = orig;
        return (lPlus - lMinus) / (2.0 * EPS);
    }

    private static double numericGradN(PowerLevelModel model, int[] tokens, int target) {
        double orig = model.n();
        model.setN(orig + EPS);
        model.forward(tokens);
        double lPlus = model.crossEntropyLoss(target);
        model.setN(orig - EPS);
        model.forward(tokens);
        double lMinus = model.crossEntropyLoss(target);
        model.setN(orig);
        return (lPlus - lMinus) / (2.0 * EPS);
    }

    private static CheckResult check(String label, double numeric, double analytic) {
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
        return new CheckResult(pass, absErr, relErr);
    }

    private record CheckResult(boolean pass, double absErr, double relErr) {}
}
