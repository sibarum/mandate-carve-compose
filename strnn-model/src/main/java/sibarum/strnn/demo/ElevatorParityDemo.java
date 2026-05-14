package sibarum.strnn.demo;

import sibarum.strnn.ksq.elevator.ElevatorAnchors;
import sibarum.strnn.ksq.elevator.ElevatorModel;

/**
 * XOR parity sweep on the elevator KSQ. Exercises the final iter-6 state
 * of {@link ElevatorModel}: tanh removed, α = sumLogits flows directly
 * into the bilinear step (Phase 2), with the 5-anchor set including
 * K_eMinus (Phase 3). Mirrors iter-5's {@link KsqParityDemo} structure
 * exactly so the comparison is clean.
 *
 * <p>Empirical: at LR=0.1 the architecture stabilizes (Phase 3 result),
 * solving XOR at 10/10 across the (λ, ν) grid with non-trivial basins.
 * Reports raw / non-trivial solve rates (trivial = both tokens K_0
 * dominant with all other |α| &lt; 0.5), same metric shape as iter 5.
 *
 * <p>The Phase 1 (unit-norm + magnitude-to-head) and Phase 2 (α = ℓ
 * unbounded, single bilinear step, no K_eMinus) intermediate stages
 * are documented in {@code docs/16-ksq-substrate.md}; only the final
 * Phase 3 state is preserved in code.
 */
public final class ElevatorParityDemo {

    private static final int VOCAB = 2;
    private static final int T = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 0.1;
    private static final double EMBED_INIT_BOUND = 2.0;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };
    private static final double[] LAMBDAS = { 0.0, 0.1, 1.0 };
    private static final double[] NUS = { 0.0, 0.1 };

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

        System.out.println("=== Elevator XOR (Phase 1: unit-norm direction + magnitude-to-head) ===");
        System.out.println("  T=" + T + ", N=" + n + ", lr=" + LR + ", epochs=" + EPOCHS
                + ", initBound=" + EMBED_INIT_BOUND + ", seeds=" + SEEDS.length);
        System.out.println("  Called-shot: results degrade vs iter-5 (tanh removed, magnitude passed to head).");
        System.out.println();
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

    private static int[] runSweep(int[][] xs, int[] ys, double lambda, double nu) {
        int n = xs.length;
        System.out.println();
        System.out.println("--- λ = " + lambda + ", ν = " + nu + " ---");
        System.out.printf("  %5s | %-5s | %-9s | %-9s | %-9s | %-10s | %-22s | %-22s%n",
                "seed", "acc", "ce", "reg", "cross", "‖ℓ‖ avg", "token 0 -> dominant", "token 1 -> dominant");
        System.out.println("  -----+-------+-----------+-----------+-----------+------------+------------------------+------------------------");

        int solved = 0;
        int nonTrivialSolved = 0;

        for (long seed : SEEDS) {
            ElevatorModel model = new ElevatorModel(VOCAB, OUT_DIM, seed, EMBED_INIT_BOUND);
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
            double mag0 = magnitudeOf(model.embedding().lookup(0));
            double mag1 = magnitudeOf(model.embedding().lookup(1));
            double magAvg = 0.5 * (mag0 + mag1);
            int dom0 = argmaxAbs(alpha0);
            int dom1 = argmaxAbs(alpha1);

            System.out.printf("  %5d | %d/%-3d | %.5f  | %.5f  | %.5f  | %.4e | %s α=%+.2f             | %s α=%+.2f%n",
                    seed, correct, n, ceLoss, regLoss, crossLoss, magAvg,
                    ElevatorAnchors.name(dom0), alpha0[dom0],
                    ElevatorAnchors.name(dom1), alpha1[dom1]);

            if (correct == n) {
                solved++;
                if (!(isScalarSpecialized(alpha0) && isScalarSpecialized(alpha1))) {
                    nonTrivialSolved++;
                }
            }
        }

        System.out.println();
        System.out.println("  λ=" + lambda + " ν=" + nu + " summary: raw=" + solved + "/" + SEEDS.length
                + "  non-trivial=" + nonTrivialSolved + "/" + SEEDS.length);
        return new int[] { solved, nonTrivialSolved };
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

    private static boolean isScalarSpecialized(double[] alpha) {
        if (argmaxAbs(alpha) != ElevatorAnchors.K0) return false;
        for (int i = 0; i < alpha.length; i++) {
            if (i == ElevatorAnchors.K0) continue;
            if (Math.abs(alpha[i]) >= ElevatorModel.SCALAR_OTHER_THRESHOLD) return false;
        }
        return true;
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
