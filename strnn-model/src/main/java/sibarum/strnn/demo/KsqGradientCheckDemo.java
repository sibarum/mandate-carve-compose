package sibarum.strnn.demo;

import sibarum.strnn.ksq.KsqAnchors;
import sibarum.strnn.ksq.KsqEmbeddingTable;
import sibarum.strnn.ksq.KsqModel;
import sibarum.strnn.ksq.KsqOutputHead;

/**
 * Finite-difference gradient check for the KSQ backward pass. The risky piece
 * of KSQ is the non-commutative matmul-chain backward; this demo verifies it
 * (and every other gradient pathway) by central-difference numerical
 * derivatives.
 *
 * <p>Each learned scalar is perturbed by ±eps; the resulting (L(+eps) -
 * L(-eps)) / (2·eps) is compared to the analytical gradient produced by
 * {@link KsqModel#backward(int)}. Failure on any parameter means a math/sign/
 * transpose bug somewhere — the diagnostic prints which param disagreed and
 * by how much.
 *
 * <p>Run BEFORE wiring a training loop. If this demo passes, the gradients
 * are correct on a tiny instance; that does not guarantee numerical stability
 * at longer T, but it does rule out a class of implementation bugs that
 * would otherwise make every downstream result a guessing game.
 */
public final class KsqGradientCheckDemo {

    private static final int VOCAB = 4;
    private static final int OUT_DIM = 3;
    private static final long SEED = 42L;

    private static final double EPS = 1e-5;
    private static final double TOL_ABS = 1e-6;
    private static final double TOL_REL = 1e-4;

    public static void main(String[] args) {
        int[] tokens = { 0, 1, 2 };
        int target = 1;

        KsqModel model = new KsqModel(VOCAB, OUT_DIM, SEED);

        model.zeroGrad();
        model.forward(tokens);
        model.backward(target);

        int totalChecked = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worstParam = "";

        // 1. Embedding table gradients (vocab x n_anchors), but we only need
        //    to check rows that were actually visited by the forward (others
        //    must have exactly zero gradient).
        KsqEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < KsqAnchors.N; a++) {
                double numeric = numericGradEmbedding(model, tokens, target, v, a);
                String label = "E[" + v + "][" + a + "]";
                CheckResult r = check(label, numeric, analyticGrad[a]);
                totalChecked++;
                if (!r.pass) totalMismatch++;
                if (r.absErr > maxAbsErr) { maxAbsErr = r.absErr; worstParam = label; }
                if (r.relErr > maxRelErr) maxRelErr = r.relErr;
            }
        }

        // 2. Output head weights (outDim x n_anchors) and biases (outDim).
        KsqOutputHead head = model.head();
        double[][] gw = head.gradWeights();
        for (int k = 0; k < OUT_DIM; k++) {
            for (int i = 0; i < KsqAnchors.N; i++) {
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

        // 3. Unvisited embedding rows must have zero gradient (structural check).
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
        System.out.println("=== KSQ gradient check ===");
        System.out.println("  T=" + tokens.length + ", vocab=" + VOCAB
                + ", n_anchors=" + KsqAnchors.N + ", outDim=" + OUT_DIM
                + ", eps=" + EPS);
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
        if (!regPassed) {
            System.exit(1);
        }

        boolean crossPassed = checkCrossVocabRegularizer();
        if (!crossPassed) {
            System.exit(1);
        }
    }

    /**
     * Standalone gradient check for the subalgebra-specialization regularizer.
     * The regularizer is computed from the embedding alone (no chain, no
     * head), so it's structurally simpler than the main backward — but if
     * the math is wrong it shows up here in isolation, not tangled with CE.
     */
    private static boolean checkRegularizer() {
        double lambda = 0.7;
        KsqModel model = new KsqModel(VOCAB, OUT_DIM, SEED ^ 0xDEADBEEFL);
        model.zeroGrad();
        model.regularizerBackward(lambda);

        int checked = 0;
        int mismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worst = "";

        KsqEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < KsqAnchors.N; a++) {
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
        System.out.println("=== KSQ regularizer gradient check (λ=" + lambda + ") ===");
        System.out.println("  checked: " + checked + " parameters");
        System.out.println("  mismatches: " + mismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst: " + worst + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean pass = (mismatch == 0);
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        return pass;
    }

    private static double numericGradEmbedding(KsqModel model, int[] tokens, int target,
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

    private static double numericGradHeadWeight(KsqModel model, int[] tokens, int target,
                                                 int outIdx, int anchorIdx) {
        double[][] w = model.head().weights();
        double orig = w[outIdx][anchorIdx];
        w[outIdx][anchorIdx] = orig + EPS;
        model.forward(tokens);
        double lPlus = model.crossEntropyLoss(target);
        w[outIdx][anchorIdx] = orig - EPS;
        model.forward(tokens);
        double lMinus = model.crossEntropyLoss(target);
        w[outIdx][anchorIdx] = orig;
        return (lPlus - lMinus) / (2.0 * EPS);
    }

    private static double numericGradHeadBias(KsqModel model, int[] tokens, int target, int outIdx) {
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

    /**
     * Standalone gradient check for the cross-vocab contrastive regularizer.
     * Like {@link #checkRegularizer()} this is computed from the embedding
     * alone, so a math bug shows up cleanly in isolation. The interesting
     * structural property: gradient flows BETWEEN vocab rows (∂R/∂α_a
     * depends on every other α_b), so we must perturb each (v, a) entry and
     * confirm the analytical gradient picks up the correct cross-token
     * influence.
     */
    private static boolean checkCrossVocabRegularizer() {
        double nu = 0.5;
        KsqModel model = new KsqModel(VOCAB, OUT_DIM, SEED ^ 0xCAFEBABEL);
        model.zeroGrad();
        model.crossVocabRegularizerBackward(nu);

        int checked = 0;
        int mismatch = 0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        String worst = "";

        KsqEmbeddingTable emb = model.embedding();
        for (int v = 0; v < VOCAB; v++) {
            double[] analyticGrad = emb.gradient(v).clone();
            for (int a = 0; a < KsqAnchors.N; a++) {
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
        System.out.println("=== KSQ cross-vocab regularizer gradient check (ν=" + nu + ") ===");
        System.out.println("  checked: " + checked + " parameters");
        System.out.println("  mismatches: " + mismatch);
        System.out.println("  max abs err: " + String.format("%.3e", maxAbsErr)
                + "   (worst: " + worst + ")");
        System.out.println("  max rel err: " + String.format("%.3e", maxRelErr));
        boolean pass = (mismatch == 0);
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        return pass;
    }

    private record CheckResult(boolean pass, double absErr, double relErr) {}
}
