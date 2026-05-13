package sibarum.strnn.demo;

import sibarum.strnn.ksq.KsqAnchors;
import sibarum.strnn.ksq.KsqModel;

/**
 * KSQ first test task: XOR (binary parity at T=2). Vocabulary {0, 1}, four
 * inputs (00, 01, 10, 11), label = (#1s) mod 2.
 *
 * <p>Why T=2 rather than T=4? Under the corrected architecture (sum-pool
 * token logits → single α → Q → Q² → β → linear head), sequence info
 * collapses to a 1D curve in α-space parameterized by count k. A linear
 * readout on Q² (quadratic in α) cannot classify the parity-of-count
 * function for k ∈ {0,1,2,3,4} because that's a sawtooth on a 1D curve,
 * not a quadratic. T=2 XOR <i>is</i> solvable: Q² introduces exactly the
 * cross-term in α that XOR needs.
 *
 * <p>Under signed α (tanh, not softmax), β_i = 2 α_0 α_i is sign-controlled,
 * which restores elliptic/hyperbolic symmetry. Predicted: BOTH K_i and K_j
 * basins should win with comparable frequency. If they do, the architecture
 * is symmetric between rotation-generator and boost-generator specialization;
 * if K_j still dominates, there's a deeper asymmetry beyond α-positivity.
 *
 * <p>Sweeps λ ∈ {0, 0.01, 0.1, 1.0} × 10 seeds to characterize the basin
 * distribution and the regularizer's effect.
 */
public final class KsqParityDemo {

    private static final int VOCAB = 2;
    private static final int T = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 1.0;
    private static final double EMBED_INIT_BOUND = 2.0;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };
    private static final double[] LAMBDAS = { 0.0, 0.1, 1.0 };
    private static final double[] NUS = { 0.0, 0.01, 0.1, 1.0 };

    public static void main(String[] args) {
        int n = 1 << T;
        int[][] xs = new int[n][T];
        int[] ys = new int[n];
        for (int s = 0; s < n; s++) {
            int count = 0;
            for (int b = 0; b < T; b++) {
                int bit = (s >> b) & 1;
                xs[s][b] = bit;
                count += bit;
            }
            ys[s] = count & 1;
        }

        System.out.println("=== KSQ XOR (binary parity, T=" + T + ", N=" + n + ") "
                + "lr=" + LR + " epochs=" + EPOCHS + " initBound=" + EMBED_INIT_BOUND
                + " seeds=" + SEEDS.length + " ===");
        System.out.println("=== (λ, ν) sweep: per-vocab + cross-vocab regularizers ===");

        int[][] solveGrid = new int[NUS.length][LAMBDAS.length];
        int[][] nonTrivialGrid = new int[NUS.length][LAMBDAS.length];

        for (int ni = 0; ni < NUS.length; ni++) {
            for (int li = 0; li < LAMBDAS.length; li++) {
                int[] result = runSweep(xs, ys, LAMBDAS[li], NUS[ni]);
                solveGrid[ni][li] = result[0];
                nonTrivialGrid[ni][li] = result[1];
            }
        }

        System.out.println();
        System.out.println("=== Solve-rate grid (raw / non-trivial, both out of 10) ===");
        System.out.println("    raw = correct logits on all 4 inputs");
        System.out.println("    non-trivial = raw AND not both tokens dominant on K_0 (avoids the");
        System.out.println("                  algebra-collapses-to-scalar trivial solution)");
        System.out.println();
        System.out.print("            ");
        for (double lambda : LAMBDAS) System.out.printf("  λ=%-12.2f", lambda);
        System.out.println();
        for (int ni = 0; ni < NUS.length; ni++) {
            System.out.printf("  ν=%-5.2f ", NUS[ni]);
            for (int li = 0; li < LAMBDAS.length; li++) {
                System.out.printf("  %2d / %-2d        ", solveGrid[ni][li], nonTrivialGrid[ni][li]);
            }
            System.out.println();
        }
    }

    /** Returns {raw_solved, non_trivial_solved}. */
    private static int[] runSweep(int[][] xs, int[] ys, double lambda, double nu) {
        int n = xs.length;
        System.out.println();
        System.out.println("--- λ = " + lambda + ", ν = " + nu + " ---");
        System.out.printf("  %5s | %-5s | %-9s | %-9s | %-9s | %-22s | %-22s | %s%n",
                "seed", "acc", "ce", "reg", "cross", "token 0 -> dominant", "token 1 -> dominant", "subalgebra");
        System.out.println("  -----+-------+-----------+-----------+-----------+------------------------+------------------------+-----------");

        int solved = 0;
        int nonTrivialSolved = 0;
        int trivialSolved = 0;
        int[] basinCount = new int[KsqAnchors.N * KsqAnchors.N];

        for (long seed : SEEDS) {
            KsqModel model = new KsqModel(VOCAB, OUT_DIM, seed, EMBED_INIT_BOUND);
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                model.zeroGrad();
                for (int s = 0; s < n; s++) {
                    model.forward(xs[s]);
                    model.backward(ys[s]);
                }
                model.regularizerBackward(lambda);
                model.crossVocabRegularizerBackward(nu);
                model.step(LR / n);
            }

            int correct = countCorrect(model, xs, ys);
            double ceLoss = epochLoss(model, xs, ys);
            double regLoss = model.regularizerLoss(lambda);
            double crossLoss = model.crossVocabRegularizerLoss(nu);

            double[] alpha0 = perTokenAlpha(model, 0);
            double[] alpha1 = perTokenAlpha(model, 1);
            int dom0 = argmaxAbs(alpha0);
            int dom1 = argmaxAbs(alpha1);
            boolean isTrivial = isScalarSpecialized(alpha0) && isScalarSpecialized(alpha1);
            String subalgebra = subalgebraLabel(alpha0, alpha1, dom0, dom1);

            System.out.printf("  %5d | %d/%-3d | %.5f  | %.5f  | %.5f  | %s α=%+.2f             | %s α=%+.2f             | %s%n",
                    seed, correct, n, ceLoss, regLoss, crossLoss,
                    KsqAnchors.name(dom0), alpha0[dom0],
                    KsqAnchors.name(dom1), alpha1[dom1],
                    subalgebra);

            if (correct == n) {
                solved++;
                if (isTrivial) trivialSolved++; else nonTrivialSolved++;
            }
            basinCount[dom0 * KsqAnchors.N + dom1]++;
        }

        System.out.println();
        System.out.println("  λ=" + lambda + " ν=" + nu + " summary:");
        System.out.println("    raw solved          : " + solved + "/" + SEEDS.length);
        System.out.println("    non-trivial solved  : " + nonTrivialSolved + "/" + SEEDS.length
                + "  (algebra actually used)");
        System.out.println("    trivial solved      : " + trivialSolved + "/" + SEEDS.length
                + "  (both tokens on K_0 — scalar collapse)");
        System.out.print("  basins: ");
        boolean first = true;
        for (int a0 = 0; a0 < KsqAnchors.N; a0++) {
            for (int a1 = 0; a1 < KsqAnchors.N; a1++) {
                int c = basinCount[a0 * KsqAnchors.N + a1];
                if (c > 0) {
                    if (!first) System.out.print(", ");
                    first = false;
                    System.out.print(KsqAnchors.name(a0) + "->" + KsqAnchors.name(a1) + ":" + c);
                }
            }
        }
        System.out.println();
        return new int[] { solved, nonTrivialSolved };
    }

    /**
     * Label the subalgebra structure of a token. Subalgebra mapping:
     * K_0 → R, K_i → C, K_j and K_inf → R[j] (iter-4 algebraic alignment).
     * A token tagged "R" is further marked as truly scalar-specialized
     * iff its other anchor components are below the threshold; otherwise
     * it's "saturated-everywhere with K_0 nominally dominant."
     */
    private static String subalgebraLabel(double[] alpha0, double[] alpha1, int dom0, int dom1) {
        String s0 = subalgebraOf(alpha0, dom0);
        String s1 = subalgebraOf(alpha1, dom1);
        if (isScalarSpecialized(alpha0) && isScalarSpecialized(alpha1)) {
            return "R x R (trivial — scalar collapse)";
        }
        return s0 + " x " + s1;
    }

    private static String subalgebraOf(double[] alpha, int dom) {
        if (dom == KsqAnchors.K0) {
            return isScalarSpecialized(alpha) ? "R" : "(sat)";
        }
        if (dom == KsqAnchors.KI) return "C";
        return "R[j]";
    }

    /**
     * A token is "scalar-specialized" if K_0 is dominant AND every other
     * anchor's coefficient is below {@link KsqModel#SCALAR_OTHER_THRESHOLD}
     * in magnitude. This rules out the "saturated everywhere with K_0
     * happening to be argmax" failure mode of the simpler check.
     */
    private static boolean isScalarSpecialized(double[] alpha) {
        if (argmaxAbs(alpha) != KsqAnchors.K0) return false;
        for (int i = 0; i < alpha.length; i++) {
            if (i == KsqAnchors.K0) continue;
            if (Math.abs(alpha[i]) >= sibarum.strnn.ksq.KsqModel.SCALAR_OTHER_THRESHOLD) {
                return false;
            }
        }
        return true;
    }

    /**
     * Per-token (per-vocab-entry) α: tanh of the embedding row for
     * {@code tokenId}. With signed α, the dominant anchor is read by
     * argmax|α[i]|, and the sign of α[dom] carries information.
     */
    private static double[] perTokenAlpha(KsqModel model, int tokenId) {
        double[] logits = model.embedding().lookup(tokenId);
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) out[i] = Math.tanh(logits[i]);
        return out;
    }

    private static int argmaxAbs(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[best])) best = i;
        return best;
    }

    private static double epochLoss(KsqModel model, int[][] xs, int[] ys) {
        double s = 0.0;
        for (int i = 0; i < xs.length; i++) {
            model.forward(xs[i]);
            s += model.crossEntropyLoss(ys[i]);
        }
        return s / xs.length;
    }

    private static int countCorrect(KsqModel model, int[][] xs, int[] ys) {
        int c = 0;
        for (int i = 0; i < xs.length; i++) {
            double[] logits = model.forward(xs[i]);
            if (argmax(logits) == ys[i]) c++;
        }
        return c;
    }

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
