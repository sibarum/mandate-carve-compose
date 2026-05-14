package sibarum.strnn.demo;

import sibarum.strnn.ksq.KsqAnchors;
import sibarum.strnn.ksq.KsqModel;

/**
 * Phase 0 of iter 6 (elevator plan): measure ρ_∞(t) under the existing
 * iter-5 KSQ architecture. The elevator plan hypothesizes that the
 * gradient is *trying* to push embedding logits toward the parabolic
 * anchor (K_∞) but tanh has been preventing the accumulation. This
 * demo measures whether the gradient is in fact asking for the elevator
 * — before committing to the architectural rewrite.
 *
 * <p>The measurement: for each parameter update step t, read the
 * embedding gradient at the K_∞ slot for each vocab row. The delta
 * applied that step is $\Delta\ell(v)[\text{KINF}] = -(\text{lr}/N)
 * \cdot \text{grad}[v][\text{KINF}]$. The inner product with the
 * canonical $K_\infty^{\text{logits}}$ direction (basis vector at the
 * KINF slot) is just this scalar.
 *
 * <p>Three outcome classifications:
 * <ul>
 *   <li>A: systematically positive during specific phases → the gradient
 *       is asking for the elevator, tanh is the lid. Proceed with confidence.</li>
 *   <li>B: oscillates around zero → either XOR doesn't need higher degree
 *       or tanh has suppressed the dynamic. Proceed with adjusted expectations.</li>
 *   <li>C: unexpected structure (spikes, negative excursions, etc.) →
 *       the measurement itself is the value; interpretation drives Phase 1.</li>
 * </ul>
 *
 * <p>No model changes. Instrumentation only. ~30 minutes to read.
 */
public final class KsqRhoInfDiagnosticDemo {

    private static final int VOCAB = 2;
    private static final int T = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 1.0;
    private static final double EMBED_INIT_BOUND = 2.0;
    private static final double LAMBDA = 0.1;
    private static final double NU = 0.1;
    private static final long[] SEEDS = { 1L, 3L, 7L, 13L, 23L };
    private static final int CHECKPOINT_EVERY = 200;

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

        System.out.println("=== KSQ ρ_∞(t) diagnostic on iter-5 architecture ===");
        System.out.println("  Setup: T=" + T + ", N=" + n + ", lr=" + LR + ", epochs=" + EPOCHS
                + ", λ=" + LAMBDA + ", ν=" + NU);
        System.out.println("  Measurement: per-epoch Δlogits[v][KINF] = -(lr/N) · grad[v][KINF]");
        System.out.println("  read after CE/reg/cross-vocab backward, before step.");
        System.out.println();

        for (long seed : SEEDS) {
            runSeed(xs, ys, seed);
        }
    }

    private static void runSeed(int[][] xs, int[] ys, long seed) {
        int n = xs.length;
        KsqModel model = new KsqModel(VOCAB, OUT_DIM, seed, EMBED_INIT_BOUND);

        // Per-token trajectory accumulators.
        double[] rhoSum = new double[VOCAB];
        double[] rhoSumAbs = new double[VOCAB];
        double[] rhoMax = new double[VOCAB];
        double[] rhoMin = new double[VOCAB];
        int[] signFlips = new int[VOCAB];
        int[] prevSign = new int[VOCAB];
        int[] maxRunSameSign = new int[VOCAB];
        int[] curRunSameSign = new int[VOCAB];
        int[] curRunSign = new int[VOCAB];
        for (int v = 0; v < VOCAB; v++) {
            rhoMax[v] = Double.NEGATIVE_INFINITY;
            rhoMin[v] = Double.POSITIVE_INFINITY;
        }

        System.out.println("--- seed " + seed + " ---");
        System.out.printf("  %6s | %12s %12s | %12s %12s%n",
                "epoch", "ρ_∞(tok0)", "cum_tok0", "ρ_∞(tok1)", "cum_tok1");

        double cum0 = 0.0;
        double cum1 = 0.0;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            model.zeroGrad();
            for (int s = 0; s < n; s++) {
                model.forward(xs[s]);
                model.backward(ys[s]);
            }
            model.regularizerBackward(LAMBDA);
            model.crossVocabRegularizerBackward(NU);

            // Measurement window: after all backward calls, before step.
            double lrPerParam = LR / n;
            double rho0 = -lrPerParam * model.embedding().gradient(0)[KsqAnchors.KINF];
            double rho1 = -lrPerParam * model.embedding().gradient(1)[KsqAnchors.KINF];

            model.step(LR / n);

            // Accumulate stats.
            cum0 += rho0;
            cum1 += rho1;
            updateStats(0, rho0, rhoSum, rhoSumAbs, rhoMax, rhoMin,
                    signFlips, prevSign, maxRunSameSign, curRunSameSign, curRunSign);
            updateStats(1, rho1, rhoSum, rhoSumAbs, rhoMax, rhoMin,
                    signFlips, prevSign, maxRunSameSign, curRunSameSign, curRunSign);

            if (epoch == 0 || (epoch + 1) % CHECKPOINT_EVERY == 0) {
                System.out.printf("  %6d | %+.4e %+.4e | %+.4e %+.4e%n",
                        epoch + 1, rho0, cum0, rho1, cum1);
            }
        }

        System.out.println();
        System.out.println("  summary:");
        for (int v = 0; v < VOCAB; v++) {
            double mean = rhoSum[v] / EPOCHS;
            double meanAbs = rhoSumAbs[v] / EPOCHS;
            System.out.printf("    token %d: mean=%+.4e  mean|·|=%.4e  max=%+.4e  min=%+.4e%n",
                    v, mean, meanAbs, rhoMax[v], rhoMin[v]);
            System.out.printf("             cumulative=%+.4e  sign_flips=%d  max_same_sign_run=%d%n",
                    v == 0 ? cum0 : cum1, signFlips[v], maxRunSameSign[v]);
        }

        // Classification hint: persistent same-sign runs > 10% of training → suggests A.
        // Sign flips > 30% of steps → suggests B.
        double persistencyThreshold = 0.10 * EPOCHS;
        double oscillationThreshold = 0.30 * EPOCHS;
        for (int v = 0; v < VOCAB; v++) {
            String hint;
            if (maxRunSameSign[v] > persistencyThreshold) {
                hint = "persistent same-sign run → outcome A (elevator asked)";
            } else if (signFlips[v] > oscillationThreshold) {
                hint = "high oscillation → outcome B (elevator not asked)";
            } else {
                hint = "mixed signal → outcome C (look at trajectory)";
            }
            System.out.println("    token " + v + " classification hint: " + hint);
        }
        System.out.println();
    }

    private static void updateStats(int v, double rho,
                                     double[] rhoSum, double[] rhoSumAbs,
                                     double[] rhoMax, double[] rhoMin,
                                     int[] signFlips, int[] prevSign,
                                     int[] maxRunSameSign, int[] curRunSameSign, int[] curRunSign) {
        rhoSum[v] += rho;
        rhoSumAbs[v] += Math.abs(rho);
        if (rho > rhoMax[v]) rhoMax[v] = rho;
        if (rho < rhoMin[v]) rhoMin[v] = rho;
        int sgn = (rho > 0) ? 1 : (rho < 0 ? -1 : 0);
        if (sgn != 0) {
            if (sgn == curRunSign[v]) {
                curRunSameSign[v]++;
            } else {
                curRunSameSign[v] = 1;
                curRunSign[v] = sgn;
            }
            if (curRunSameSign[v] > maxRunSameSign[v]) maxRunSameSign[v] = curRunSameSign[v];
            if (prevSign[v] != 0 && sgn != prevSign[v]) signFlips[v]++;
            prevSign[v] = sgn;
        }
    }
}
