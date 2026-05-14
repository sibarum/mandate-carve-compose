package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.PowerLevelModel;

/**
 * Iter 7 prediction battery: train PowerLevelModel on parity tasks of
 * increasing T and observe what the learned scalar $n$ does. The
 * predictions tested:
 * <ul>
 *   <li>T=2 XOR solves with $n$ near 1 (matches iter-5/6 baseline).</li>
 *   <li>T=4 parity solves (vs iter-6's 0/10 falsification) with $n$ near 2.</li>
 *   <li>T=8 parity solves with $n$ near 4.</li>
 * </ul>
 *
 * <p><b>Empirical result (falsifies the free-$n$ form):</b> T=2 solves
 * 10/10 with $n$ clustering near 1.2 (consistent with degree-2 task).
 * T=4 and T=8 do NOT solve — $n$ drifts *downward* (often negative)
 * or diverges to NaN / extreme magnitudes, instead of climbing toward
 * the predicted positive value. The optimizer never reaches $n \approx 2$
 * from $n_{init}=1$ through gradient descent.
 *
 * <p>The ablation demo {@link PowerLevelAblationDemo} disentangles
 * "expressivity exists" from "optimizer reaches it": with $n$ frozen
 * at 2.0 and conservative hyperparameters (embed_init=0.25, LR=0.01),
 * T=4 parity solves 10/10. So the architecture <i>can</i> express
 * degree-4 features at $n=2$; the failure mode is gradient flow not
 * reaching that region, not an expressivity ceiling.
 *
 * <p>No regularizers (clean test of the mechanism alone). $n$ initialized
 * to 1.0 (architecture starts as iter-5-equivalent); training is given
 * the chance to discover the right level via gradient descent.
 */
public final class PowerLevelParityDemo {

    private static final int VOCAB = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 0.1;
    private static final double EMBED_INIT = 1.0;
    private static final double N_INIT = 1.0;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };
    private static final int[] TS = { 2, 4, 8 };

    public static void main(String[] args) {
        System.out.println("=== PowerLevelModel — iter 7 prediction battery ===");
        System.out.println("  signed-power activation y_i = sign(s_i)·|s_i|^n  (n learnable, scalar)");
        System.out.println("  Setup: lr=" + LR + ", epochs=" + EPOCHS + ", n_init=" + N_INIT
                + ", embed_init=" + EMBED_INIT + ", no regularizers");
        System.out.println();

        for (int T : TS) {
            runDegreeSweep(T);
        }
    }

    private static void runDegreeSweep(int T) {
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

        System.out.println("--- T = " + T + " (parity over " + n + " inputs; predicted n ≈ " + T / 2 + ") ---");
        System.out.printf("  %5s | %7s | %-9s | %-9s | %-22s | %-22s%n",
                "seed", "acc", "ce", "final n", "token 0 -> dominant", "token 1 -> dominant");
        System.out.println("  -----+---------+-----------+-----------+------------------------+------------------------");

        int solved = 0;
        double[] finalNs = new double[SEEDS.length];
        int s = 0;

        for (long seed : SEEDS) {
            PowerLevelModel model = new PowerLevelModel(VOCAB, OUT_DIM, seed, EMBED_INIT, N_INIT);

            boolean diverged = false;
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                model.zeroGrad();
                for (int i = 0; i < n; i++) {
                    model.forward(xs[i]);
                    model.backward(ys[i]);
                }
                model.step(LR / n);

                if (epoch > 0 && epoch % 500 == 0) {
                    double nv = model.n();
                    if (!Double.isFinite(nv) || Math.abs(nv) > 100) {
                        diverged = true;
                        break;
                    }
                }
            }

            if (diverged) {
                System.out.printf("  %5d | DIVERGED  (final n=%+.3e)%n", seed, model.n());
                finalNs[s++] = Double.NaN;
                continue;
            }

            int correct = countCorrect(model, xs, ys);
            double ce = epochLoss(model, xs, ys);
            double finalN = model.n();
            finalNs[s] = finalN;

            double[] alpha0 = perTokenY(model, 0);
            double[] alpha1 = perTokenY(model, 1);
            int d0 = argmaxAbs(alpha0);
            int d1 = argmaxAbs(alpha1);

            System.out.printf("  %5d | %3d/%-3d | %.5f  | %+.5f  | %s y=%+.2f             | %s y=%+.2f%n",
                    seed, correct, n, ce, finalN,
                    ElevatorAnchors.name(d0), alpha0[d0],
                    ElevatorAnchors.name(d1), alpha1[d1]);

            if (correct == n) solved++;
            s++;
        }

        System.out.println();
        System.out.printf("  T=%d summary: solved %d/%d%n", T, solved, SEEDS.length);
        reportNStats(finalNs, T);
        System.out.println();
    }

    private static void reportNStats(double[] ns, int T) {
        int finite = 0;
        double sum = 0.0;
        double sumSq = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : ns) {
            if (Double.isFinite(v)) {
                finite++;
                sum += v;
                sumSq += v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        if (finite == 0) {
            System.out.println("    final n: (no finite values)");
            return;
        }
        double mean = sum / finite;
        double var = sumSq / finite - mean * mean;
        double std = Math.sqrt(Math.max(0.0, var));
        double predicted = T / 2.0;
        System.out.printf("    final n: n=%d finite  mean=%+.3f  std=%.3f  min=%+.3f  max=%+.3f  (predicted ≈ %.1f)%n",
                finite, mean, std, min, max, predicted);
    }

    private static double[] perTokenY(PowerLevelModel model, int tokenId) {
        double[] ell = model.embedding().lookup(tokenId);
        double n = model.n();
        double[] y = new double[ell.length];
        for (int i = 0; i < ell.length; i++) {
            double s = ell[i];
            y[i] = (s == 0.0) ? 0.0 : Math.signum(s) * Math.pow(Math.abs(s), n);
        }
        return y;
    }

    private static double epochLoss(PowerLevelModel model, int[][] xs, int[] ys) {
        double s = 0.0;
        for (int i = 0; i < xs.length; i++) {
            model.forward(xs[i]);
            s += model.crossEntropyLoss(ys[i]);
        }
        return s / xs.length;
    }

    private static int countCorrect(PowerLevelModel model, int[][] xs, int[] ys) {
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

    private static int argmaxAbs(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[best])) best = i;
        return best;
    }
}
