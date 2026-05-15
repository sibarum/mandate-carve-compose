package sibarum.strnn.demo;

import sibarum.strnn.hpb.HpbModel;

/**
 * Diagnostic for the iter-1 CE plateau: does 1D K=1 XOR descend to
 * near-machine-zero CE given enough epochs, or does it plateau above
 * the float64 underflow threshold?
 *
 * <p>Three predicted outcomes (probabilities from prior analysis):
 * <ul>
 *   <li><b>A (60-70%)</b>: CE drops to ~1e-13 by ~50k epochs. "Pure rate"
 *       hypothesis confirmed — 1D ray is the joint ray, 4× slower.</li>
 *   <li><b>B (20-30%)</b>: CE drops to ~1e-13 but takes much longer.
 *       Still on the same ray, but rate ratio deviates from the 4×
 *       feature-norm prediction (Hessian structure matters).</li>
 *   <li><b>C (5-15%)</b>: CE plateaus above 1e-13 persistently. 1D and
 *       joint are NOT on equivalent rays — structural geometric bound.</li>
 * </ul>
 *
 * <p>Logs CE, weight norm, and final-logit-margin at 2k, 5k, 10k, 20k,
 * 50k, 100k, 200k epochs for 3 seeds.
 */
public final class HpbXorLongRunDemo {

    private static final int K = 1;
    private static final int OUT_DIM = 2;
    private static final double LR = 0.05;
    private static final long[] SEEDS = { 1L, 2L, 3L };
    private static final int[] CHECKPOINTS = { 2000, 5000, 10000, 20000, 50000, 100000, 200000 };

    public static void main(String[] args) {
        double[] xs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        int[] ys = { 0, 1, 1, 0 };

        System.out.println();
        System.out.println("=== HPB 1D K=1 XOR — long-run CE descent ===");
        System.out.println("  K=1 (featDim=2), 6 params total");
        System.out.println("  4 training points, linearly separable in features");
        System.out.println("  lr=" + LR + ", " + SEEDS.length + " seeds");
        System.out.println();

        for (long seed : SEEDS) {
            System.out.printf("--- seed %d ---%n", seed);
            System.out.printf("  %8s | %-12s | %-9s | %-9s | %s%n",
                    "epochs", "CE", "||W||", "logit margin", "correct");

            HpbModel model = new HpbModel(K, OUT_DIM, seed);
            int prevEpoch = 0;
            for (int target : CHECKPOINTS) {
                int delta = target - prevEpoch;
                for (int e = 0; e < delta; e++) {
                    model.zeroGrad();
                    for (int i = 0; i < xs.length; i++) {
                        model.forward(xs[i]);
                        model.backward(ys[i]);
                    }
                    model.step(LR / xs.length);
                }
                prevEpoch = target;

                double ce = epochLoss(model, xs, ys);
                double wNorm = weightNorm(model);
                double margin = minLogitMargin(model, xs, ys);
                int correct = countCorrect(model, xs, ys);
                System.out.printf("  %8d | %.4e   | %8.2f | %12.3f | %d/%d%n",
                        target, ce, wNorm, margin, correct, xs.length);
            }
            System.out.println();
        }
    }

    private static double weightNorm(HpbModel m) {
        double s = 0.0;
        for (double[] row : m.weights()) for (double w : row) s += w * w;
        for (double b : m.biases()) s += b * b;
        return Math.sqrt(s);
    }

    private static double minLogitMargin(HpbModel m, double[] xs, int[] ys) {
        double minMargin = Double.POSITIVE_INFINITY;
        for (int i = 0; i < xs.length; i++) {
            double[] logits = m.forward(xs[i]);
            double rightLogit = logits[ys[i]];
            double otherLogit = logits[1 - ys[i]];
            double margin = rightLogit - otherLogit;
            if (margin < minMargin) minMargin = margin;
        }
        return minMargin;
    }

    private static double epochLoss(HpbModel m, double[] xs, int[] ys) {
        double s = 0.0;
        for (int i = 0; i < xs.length; i++) {
            m.forward(xs[i]);
            s += m.crossEntropyLoss(ys[i]);
        }
        return s / xs.length;
    }

    private static int countCorrect(HpbModel m, double[] xs, int[] ys) {
        int c = 0;
        for (int i = 0; i < xs.length; i++) {
            double[] logits = m.forward(xs[i]);
            if (argmax(logits) == ys[i]) c++;
        }
        return c;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
