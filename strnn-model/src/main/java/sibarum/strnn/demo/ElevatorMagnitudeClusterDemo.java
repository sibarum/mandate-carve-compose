package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.ElevatorModel;

/**
 * Phase 4 prediction battery: train the elevator KSQ on parity tasks
 * of increasing degree (T=2 XOR, T=4 parity, T=8 parity) and measure
 * (a) solve rate, (b) trained ‖ℓ‖ clusters per token. The elevator
 * plan's strong prediction: T=4 becomes solvable (it was provably
 * unsolvable under iter-5's tanh-bound) and ‖ℓ‖ clusters at
 * task-dependent values — degree-2 task has lower ‖ℓ‖ than degree-4,
 * which has lower than degree-8. If the relationship is monotone,
 * the magnitude-is-level hypothesis lands.
 *
 * <p>Hyperparameters: LR=0.1 (Phase 3 stable point), ν=0.1, λ=0.1.
 */
public final class ElevatorMagnitudeClusterDemo {

    private static final int VOCAB = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 0.1;
    private static final double LAMBDA = 0.1;
    private static final double NU = 0.1;
    private static final double CLIP_NORM = 10.0;
    private static final double EMBED_INIT_BOUND = 2.0;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };
    private static final int[] TS = { 2, 4, 8 };

    public static void main(String[] args) {
        System.out.println("=== Elevator KSQ — Phase 4 prediction battery ===");
        System.out.println("  Task: parity. T ∈ {2, 4, 8}. λ=" + LAMBDA + ", ν=" + NU
                + ", lr=" + LR + ", epochs=" + EPOCHS);
        System.out.println("  Predict: T=4 solvable, magnitudes cluster monotonically with T.");
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

        System.out.println("--- T = " + T + " (parity over " + n + " inputs) ---");
        System.out.printf("  %5s | %6s | %-9s | %-9s | %-9s | %-22s | %-22s%n",
                "seed", "acc", "ce", "‖ℓ₀‖", "‖ℓ₁‖", "token 0 -> dominant", "token 1 -> dominant");
        System.out.println("  -----+--------+-----------+-----------+-----------+------------------------+------------------------");

        int solved = 0;
        double[] mags0 = new double[SEEDS.length];
        double[] mags1 = new double[SEEDS.length];
        int s = 0;

        for (long seed : SEEDS) {
            ElevatorModel model = new ElevatorModel(VOCAB, OUT_DIM, seed, EMBED_INIT_BOUND);
            boolean diverged = false;
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                model.zeroGrad();
                for (int i = 0; i < n; i++) {
                    model.forward(xs[i]);
                    model.backward(ys[i]);
                }
                model.regularizerBackward(LAMBDA);
                model.crossVocabRegularizerBackward(NU);
                model.stepClipped(LR / n, CLIP_NORM);

                // Quick divergence check every 500 epochs
                if (epoch > 0 && epoch % 500 == 0) {
                    double m = magnitudeOf(model.embedding().lookup(0));
                    if (!Double.isFinite(m) || m > 1e6) {
                        diverged = true;
                        break;
                    }
                }
            }

            if (diverged) {
                System.out.printf("  %5d | DIVERGED%n", seed);
                mags0[s] = Double.NaN;
                mags1[s] = Double.NaN;
                s++;
                continue;
            }

            int correct = countCorrect(model, xs, ys);
            double ce = epochLoss(model, xs, ys);
            mags0[s] = magnitudeOf(model.embedding().lookup(0));
            mags1[s] = magnitudeOf(model.embedding().lookup(1));
            double[] a0 = perTokenAlpha(model, 0);
            double[] a1 = perTokenAlpha(model, 1);
            int d0 = argmaxAbs(a0);
            int d1 = argmaxAbs(a1);

            System.out.printf("  %5d | %3d/%-3d | %.5f  | %.3e | %.3e | %s α=%+.2f             | %s α=%+.2f%n",
                    seed, correct, n, ce, mags0[s], mags1[s],
                    ElevatorAnchors.name(d0), a0[d0],
                    ElevatorAnchors.name(d1), a1[d1]);

            if (correct == n) solved++;
            s++;
        }

        System.out.println();
        System.out.printf("  T=%d summary: solved %d/%d%n", T, solved, SEEDS.length);
        reportMagStats(mags0, "‖ℓ₀‖");
        reportMagStats(mags1, "‖ℓ₁‖");
        System.out.println();
    }

    private static void reportMagStats(double[] mags, String label) {
        int finite = 0;
        double sum = 0.0;
        double sumSq = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double m : mags) {
            if (Double.isFinite(m)) {
                finite++;
                sum += m;
                sumSq += m * m;
                if (m < min) min = m;
                if (m > max) max = m;
            }
        }
        if (finite == 0) {
            System.out.println("    " + label + ": (no finite values)");
            return;
        }
        double mean = sum / finite;
        double var = sumSq / finite - mean * mean;
        double std = Math.sqrt(Math.max(0.0, var));
        System.out.printf("    %s: n=%d  mean=%.3f  std=%.3f  min=%.3f  max=%.3f%n",
                label, finite, mean, std, min, max);
    }

    private static double[] perTokenAlpha(ElevatorModel model, int tokenId) {
        double[] logits = model.embedding().lookup(tokenId);
        double m = magnitudeOf(logits);
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) out[i] = logits[i] / m;
        return out;
    }

    private static double magnitudeOf(double[] v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return Math.sqrt(s + 1e-12);
    }

    private static double epochLoss(ElevatorModel model, int[][] xs, int[] ys) {
        double s = 0.0;
        for (int i = 0; i < xs.length; i++) {
            model.forward(xs[i]);
            s += model.crossEntropyLoss(ys[i]);
        }
        return s / xs.length;
    }

    private static int countCorrect(ElevatorModel model, int[][] xs, int[] ys) {
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
