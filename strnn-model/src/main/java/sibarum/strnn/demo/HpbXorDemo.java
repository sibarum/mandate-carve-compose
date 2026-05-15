package sibarum.strnn.demo;

import sibarum.strnn.hpb.HpbModel;

/**
 * T5 from {@code docs/harmonic_piecewise_basis.md}: train the raw
 * (delta-kernel) harmonic basis on XOR with a linear readout.
 *
 * <p>Encoding: the four XOR inputs are mapped to a single scalar
 * {@code x = (4n + 1) / 16} where {@code n = 2·b0 + b1 ∈ {0,1,2,3}},
 * placing inputs at {1/16, 5/16, 9/16, 13/16}. These positions are
 * mid-piece for both k=1 and k=2 (no breakpoint collisions). Labels are
 * the standard XOR truth table.
 *
 * <p>Why this is enough: at these four positions, sq_1(x) takes values
 * {+4, −4, −4, +4} — exactly the XOR sign pattern, scaled. A linear
 * readout with W[sq_1] = −1/4, all others 0, solves XOR before any
 * training; gradient descent should find this (or an equivalent K=2
 * solution) for any seed.
 *
 * <p>Prediction: ≥9/10 seeds solve under both K=1 and K=2.
 * F2 fires if XOR cannot be solved by the raw basis — the
 * "exact-rational basin" claim would be wrong in that case.
 */
public final class HpbXorDemo {

    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 2000;
    private static final double LR = 0.05;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };

    public static void main(String[] args) {
        double[] xs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        int[] ys = { 0, 1, 1, 0 };

        for (int K : new int[] { 1, 2 }) {
            System.out.println();
            System.out.println("=== HPB XOR (T5), K=" + K + ", featDim=" + (2 * K)
                    + ", epochs=" + EPOCHS + ", lr=" + LR + ", seeds=" + SEEDS.length + " ===");
            System.out.printf("  %5s | %3s | %-12s%n", "seed", "acc", "final CE");
            System.out.println("  ------+-----+-------------");
            int solved = 0;
            for (long seed : SEEDS) {
                HpbModel model = new HpbModel(K, OUT_DIM, seed);
                for (int epoch = 0; epoch < EPOCHS; epoch++) {
                    model.zeroGrad();
                    for (int i = 0; i < xs.length; i++) {
                        model.forward(xs[i]);
                        model.backward(ys[i]);
                    }
                    model.step(LR / xs.length);
                }
                int correct = countCorrect(model, xs, ys);
                double ce = epochLoss(model, xs, ys);
                System.out.printf("  %5d | %d/%d | %.6e%n", seed, correct, xs.length, ce);
                if (correct == xs.length) solved++;
            }
            System.out.println();
            System.out.println("  solved: " + solved + "/" + SEEDS.length);
        }
    }

    private static int countCorrect(HpbModel m, double[] xs, int[] ys) {
        int c = 0;
        for (int i = 0; i < xs.length; i++) {
            double[] logits = m.forward(xs[i]);
            if (argmax(logits) == ys[i]) c++;
        }
        return c;
    }

    private static double epochLoss(HpbModel m, double[] xs, int[] ys) {
        double s = 0.0;
        for (int i = 0; i < xs.length; i++) {
            m.forward(xs[i]);
            s += m.crossEntropyLoss(ys[i]);
        }
        return s / xs.length;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
