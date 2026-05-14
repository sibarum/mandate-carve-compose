package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.PowerLevelModel;

/**
 * Diagnostic ablation for iter 7's free-n result. The free-n battery
 * ({@link PowerLevelParityDemo}) showed T=4 and T=8 fail
 * catastrophically; n drifts downward or diverges rather than upward
 * toward the predicted 2 and 4.
 *
 * <p>This demo asks two diagnostic questions and answers both:
 * <ul>
 *   <li><b>Does the iter-6 falsification recur at n=1?</b> Frozen n=1.0
 *       on T=4 → 0/10. Recovers iter-6 Phase 4. Confirms the test is
 *       isolating the right thing.</li>
 *   <li><b>Is the expressivity there at n=2?</b> Frozen n=2.0 on T=4
 *       with default LR=0.1: diverges 10/10. With embed_init=0.25 and
 *       LR=0.01: <b>solves 10/10</b>. With LR=0.001: solves 6/10
 *       (slower convergence).</li>
 * </ul>
 *
 * <p>So the architecture <i>can</i> express degree-4 features at $n=2$;
 * the free-n battery's failure is gradient flow not <i>reaching</i>
 * $n=2$, not a missing expressivity. The signed-power activation
 * gives the right ceiling; the optimizer trajectory through $n$-space
 * is the failure mode. T=8 with n=4 remains 0/10 even at conservative
 * settings — optimization difficulty scales with the target $n$.
 *
 * <p>Frozen n means: standard SGD on embedding and head, but n is held
 * constant (no n-gradient applied). Uses {@link PowerLevelModel#stepFrozenN}.
 */
public final class PowerLevelAblationDemo {

    private static final int VOCAB = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 0.1;
    private static final double EMBED_INIT = 1.0;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };

    public static void main(String[] args) {
        System.out.println("=== PowerLevel ablation — frozen-n diagnostic ===");
        System.out.println();

        // Sanity: T=2 with n=1 should solve (iter-5 baseline behavior).
        runFrozen(2, 1.0, 1.0, 0.1);
        // Iter-6 falsification recurrence: T=4 with n=1.
        runFrozen(4, 1.0, 1.0, 0.1);
        // Expressivity test at default hyperparameters: T=4 with n=2.
        runFrozen(4, 2.0, 1.0, 0.1);
        // Magnitude-aware retest: smaller LR and embed_init for n=2 amplification.
        runFrozen(4, 2.0, 0.25, 0.01);
        runFrozen(4, 2.0, 0.25, 0.001);
        // T=8 with n=4, conservative hyperparameters.
        runFrozen(8, 4.0, 0.25, 0.001);
    }

    private static void runFrozen(int T, double nFrozen, double embedInit, double lr) {
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

        System.out.println("--- T = " + T + ",  n frozen at " + nFrozen
                + ",  embed_init=" + embedInit + ",  LR=" + lr
                + "  (predicted degree of features: " + (2 * nFrozen) + ") ---");
        System.out.printf("  %5s | %7s | %-9s | %-22s | %-22s%n",
                "seed", "acc", "ce", "token 0 -> dominant", "token 1 -> dominant");
        System.out.println("  -----+---------+-----------+------------------------+------------------------");

        int solved = 0;
        for (long seed : SEEDS) {
            PowerLevelModel model = new PowerLevelModel(VOCAB, OUT_DIM, seed, embedInit, nFrozen);

            boolean diverged = false;
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                model.zeroGrad();
                for (int i = 0; i < n; i++) {
                    model.forward(xs[i]);
                    model.backward(ys[i]);
                }
                model.stepFrozenN(lr / n);

                if (epoch > 0 && epoch % 500 == 0) {
                    double m = magnitudeOf(model.embedding().lookup(0));
                    if (!Double.isFinite(m) || m > 1e8) {
                        diverged = true;
                        break;
                    }
                }
            }

            if (diverged) {
                System.out.printf("  %5d | DIVERGED%n", seed);
                continue;
            }

            int correct = countCorrect(model, xs, ys);
            double ce = epochLoss(model, xs, ys);
            double[] y0 = perTokenY(model, 0);
            double[] y1 = perTokenY(model, 1);
            int d0 = argmaxAbs(y0);
            int d1 = argmaxAbs(y1);

            System.out.printf("  %5d | %3d/%-3d | %.5f  | %s y=%+.2f             | %s y=%+.2f%n",
                    seed, correct, n, ce,
                    ElevatorAnchors.name(d0), y0[d0],
                    ElevatorAnchors.name(d1), y1[d1]);

            if (correct == n) solved++;
        }

        System.out.println("  → solved " + solved + "/" + SEEDS.length);
        System.out.println();
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

    private static double magnitudeOf(double[] v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return Math.sqrt(s + 1e-12);
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
