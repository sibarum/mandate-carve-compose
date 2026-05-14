package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.ElevatorEmbeddingTable;
import sibarum.strnn.ksq.elevator.ElevatorModel;
import sibarum.strnn.ksq.elevator.ElevatorOutputHead;

/**
 * Finite-difference gradient check for the elevator KSQ. Phase 1 of
 * iter 6 adds a new backward pathway: unit normalization (α = ℓ / ||ℓ||)
 * with the standard L2-normalize Jacobian projection. This demo
 * verifies the math before any training, in the same shape as iter-5's
 * {@link KsqGradientCheckDemo}: every gradient pathway tested at
 * machine epsilon, with separate checks for CE+chain backward,
 * per-vocab regularizer backward, and cross-vocab regularizer
 * backward.
 *
 * <p>If any pathway disagrees with finite differences, the math is
 * wrong and any subsequent training behavior would be uninterpretable.
 */
public final class ElevatorGradientCheckDemo {

    private static final int VOCAB = 4;
    private static final int OUT_DIM = 3;
    private static final long SEED = 42L;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    public static void main(String[] args) {
        int[] tokens = { 0, 1, 2 };
        int target = 1;

        ElevatorModel model = new ElevatorModel(VOCAB, OUT_DIM, SEED);

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

        int unvisitedNonZero = 0;
        for (int v = 0; v < VOCAB; v++) {
            boolean visited = false;
            for (int t : tokens) if (t == v) { visited = true; break; }
            if (visited) continue;
            for (double g : emb.gradient(v)) {
                if (Math.abs(g) > 0.0) unvisitedNonZero++;
            }
        }

        System.out.println();
        System.out.println("=== Elevator gradient check (Phase 1: unit-norm + magnitude-to-head) ===");
        System.out.println("  T=" + tokens.length + ", vocab=" + VOCAB
                + ", n_anchors=" + ElevatorAnchors.N + ", outDim=" + OUT_DIM
                + ", head_input=" + head.inputDim() + ", eps=" + EPS);
        System.out.println("  checked: " + totalChecked + " parameters");
        System.out.println("  mismatches: " + totalMismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst param: " + worstParam + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        System.out.println("  unvisited rows with non-zero grad: " + unvisitedNonZero
                + "   (must be 0)");
        boolean passed = (totalMismatch == 0) && (unvisitedNonZero == 0);
        System.out.println("  RESULT: " + (passed ? "PASS" : "FAIL"));
        if (!passed) {
            System.exit(1);
        }

        boolean regPassed = checkRegularizer();
        if (!regPassed) System.exit(1);

        boolean crossPassed = checkCrossVocabRegularizer();
        if (!crossPassed) System.exit(1);
    }

    private static double numericGradEmbedding(ElevatorModel model, int[] tokens, int target,
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

    private static double numericGradHeadWeight(ElevatorModel model, int[] tokens, int target,
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

    private static double numericGradHeadBias(ElevatorModel model, int[] tokens, int target, int outIdx) {
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

    private static boolean checkRegularizer() {
        double lambda = 0.7;
        ElevatorModel model = new ElevatorModel(VOCAB, OUT_DIM, SEED ^ 0xDEADBEEFL);
        model.zeroGrad();
        model.regularizerBackward(lambda);

        int checked = 0;
        int mismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worst = "";

        ElevatorEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < ElevatorAnchors.N; a++) {
                double[] row = emb.lookup(v);
                double orig = row[a];
                row[a] = orig + EPS;
                double lPlus = model.regularizerLoss(lambda);
                row[a] = orig - EPS;
                double lMinus = model.regularizerLoss(lambda);
                row[a] = orig;
                double numeric = (lPlus - lMinus) / (2.0 * EPS);

                String label = "regE[" + v + "][" + a + "]";
                double absErr = Math.abs(numeric - analyticGrad[a]);
                double denom = Math.max(1.0, Math.max(Math.abs(numeric), Math.abs(analyticGrad[a])));
                double relErr = absErr / denom;
                boolean pass = (absErr < TOL_ABS) || (relErr < TOL_REL);
                if (!pass) {
                    mismatch++;
                    System.out.println("  REG MISMATCH " + label
                            + " analytic=" + String.format("%+.6e", analyticGrad[a])
                            + " numeric=" + String.format("%+.6e", numeric));
                }
                if (absErr > maxAbsErr) { maxAbsErr = absErr; worst = label; }
                if (relErr > maxRelErr) maxRelErr = relErr;
                checked++;
            }
        }

        System.out.println();
        System.out.println("=== Elevator per-vocab regularizer gradient check (λ=" + lambda + ") ===");
        System.out.println("  checked: " + checked + " parameters");
        System.out.println("  mismatches: " + mismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst: " + worst + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean pass = (mismatch == 0);
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        return pass;
    }

    private static boolean checkCrossVocabRegularizer() {
        double nu = 0.5;
        ElevatorModel model = new ElevatorModel(VOCAB, OUT_DIM, SEED ^ 0xCAFEBABEL);
        model.zeroGrad();
        model.crossVocabRegularizerBackward(nu);

        int checked = 0;
        int mismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worst = "";

        ElevatorEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < ElevatorAnchors.N; a++) {
                double[] row = emb.lookup(v);
                double orig = row[a];
                row[a] = orig + EPS;
                double lPlus = model.crossVocabRegularizerLoss(nu);
                row[a] = orig - EPS;
                double lMinus = model.crossVocabRegularizerLoss(nu);
                row[a] = orig;
                double numeric = (lPlus - lMinus) / (2.0 * EPS);

                String label = "crossE[" + v + "][" + a + "]";
                double absErr = Math.abs(numeric - analyticGrad[a]);
                double denom = Math.max(1.0, Math.max(Math.abs(numeric), Math.abs(analyticGrad[a])));
                double relErr = absErr / denom;
                boolean pass = (absErr < TOL_ABS) || (relErr < TOL_REL);
                if (!pass) {
                    mismatch++;
                    System.out.println("  CROSS MISMATCH " + label
                            + " analytic=" + String.format("%+.6e", analyticGrad[a])
                            + " numeric=" + String.format("%+.6e", numeric));
                }
                if (absErr > maxAbsErr) { maxAbsErr = absErr; worst = label; }
                if (relErr > maxRelErr) maxRelErr = relErr;
                checked++;
            }
        }

        System.out.println();
        System.out.println("=== Elevator cross-vocab regularizer gradient check (ν=" + nu + ") ===");
        System.out.println("  checked: " + checked + " parameters");
        System.out.println("  mismatches: " + mismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst: " + worst + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean pass = (mismatch == 0);
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        return pass;
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
