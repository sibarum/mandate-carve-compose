package sibarum.strnn.demo;

import sibarum.strnn.ksq.KsqAnchors;
import sibarum.strnn.ksq.KsqModel;

/**
 * Saddle-vs-basin test for KSQ. Initialize the embedding directly at the
 * scalar-trivial configuration — token 0 → pure +K_0, token 1 → pure -K_0,
 * other anchor logits = 0 — and run training. Does the optimizer LEAVE
 * the trivial configuration to find an algebraically richer one?
 *
 * <p>If YES (scalar-trivial is a saddle): "the architecture prefers to use
 * its substrate" is a confirmed dynamical claim about training, not just an
 * observation about random-initialization outcomes. The
 * {@link KsqParityDemo} finding that 0/120 random-init trials trivialize
 * gets a complementary explanation: even when started at trivial, training
 * moves away.
 *
 * <p>If NO (scalar-trivial is a basin): trivial is a stable attractor; the
 * KsqParityDemo absence of trivial outcomes is initialization bias, not
 * architectural preference. The substrate-as-primitive framing weakens.
 *
 * <p>Three init magnitudes test the role of tanh saturation:
 * <ul>
 *   <li>M = 0.5 (α ≈ ±0.46, tanh' ≈ 0.79): gradient flows freely</li>
 *   <li>M = 1.0 (α ≈ ±0.76, tanh' ≈ 0.42): gradient flows but attenuated</li>
 *   <li>M = 3.0 (α ≈ ±0.995, tanh' ≈ 0.01): essentially saturated</li>
 * </ul>
 * At hard saturation the test is uninformative (no gradient regardless of
 * substrate preference); only the softer inits reveal the basin/saddle
 * distinction.
 */
public final class KsqScalarTrivialDemo {

    private static final int VOCAB = 2;
    private static final int T = 2;
    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 4000;
    private static final double LR = 1.0;
    private static final long SEED = 7L;

    private static final double[] INIT_MAGS = { 0.5, 1.0, 3.0 };

    private static final double[][] CONDITIONS = {
            { 0.0, 0.0 },
            { 0.1, 0.0 },
            { 0.0, 0.1 },
            { 0.1, 0.1 },
    };
    private static final String[] COND_NAMES = {
            "CE only",
            "per-vocab λ=0.1",
            "cross-vocab ν=0.1",
            "both λ=0.1 ν=0.1",
    };

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

        System.out.println("=== KSQ Scalar-Trivial Saddle Test ===");
        System.out.println("Init: token 0 logits = (+M, 0, 0, 0), token 1 logits = (-M, 0, 0, 0).");
        System.out.println("Train under various regularizers; does α leave the K_0-only configuration?");
        System.out.println();

        for (double initMag : INIT_MAGS) {
            System.out.println("=== Init magnitude M = " + initMag
                    + "  (tanh(M) = " + String.format("%.3f", Math.tanh(initMag))
                    + ",  tanh'(M) = " + String.format("%.3f", 1.0 - Math.tanh(initMag) * Math.tanh(initMag))
                    + ") ===");
            for (int c = 0; c < CONDITIONS.length; c++) {
                runTrial(xs, ys, CONDITIONS[c][0], CONDITIONS[c][1], initMag, COND_NAMES[c]);
            }
            System.out.println();
        }
    }

    private static void runTrial(int[][] xs, int[] ys, double lambda, double nu,
                                  double initMag, String label) {
        int n = xs.length;
        KsqModel model = new KsqModel(VOCAB, OUT_DIM, SEED, 0.0);
        double[] logits0 = model.embedding().lookup(0);
        logits0[0] = +initMag;
        for (int i = 1; i < KsqAnchors.N; i++) logits0[i] = 0.0;
        double[] logits1 = model.embedding().lookup(1);
        logits1[0] = -initMag;
        for (int i = 1; i < KsqAnchors.N; i++) logits1[i] = 0.0;

        double[] alpha0Init = perTokenAlpha(model, 0);
        double[] alpha1Init = perTokenAlpha(model, 1);

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

        double[] alpha0Final = perTokenAlpha(model, 0);
        double[] alpha1Final = perTokenAlpha(model, 1);
        int correct = countCorrect(model, xs, ys);

        double maxMove0 = maxComponentMagnitude(alpha0Final, KsqAnchors.K0);
        double maxMove1 = maxComponentMagnitude(alpha1Final, KsqAnchors.K0);
        double maxMove = Math.max(maxMove0, maxMove1);
        String verdict;
        if (maxMove < 0.01) verdict = "STAYED at scalar";
        else if (maxMove < 0.1) verdict = "barely moved";
        else verdict = "LEFT scalar";

        System.out.printf("  %-20s | acc=%d/%d | %-16s | max non-K_0 |α|: tok0=%.3f tok1=%.3f%n",
                label, correct, n, verdict, maxMove0, maxMove1);
        System.out.print("    α(0)init = [");
        printVec(alpha0Init);
        System.out.print(" ]  →  α(0)final = [");
        printVec(alpha0Final);
        System.out.println(" ]");
        System.out.print("    α(1)init = [");
        printVec(alpha1Init);
        System.out.print(" ]  →  α(1)final = [");
        printVec(alpha1Final);
        System.out.println(" ]");
    }

    private static void printVec(double[] v) {
        for (double x : v) System.out.printf(" %+.3f", x);
    }

    /** Largest |α[i]| over indices i ≠ excludeIndex. Measures "departure from scalar." */
    private static double maxComponentMagnitude(double[] alpha, int excludeIndex) {
        double m = 0.0;
        for (int i = 0; i < alpha.length; i++) {
            if (i == excludeIndex) continue;
            if (Math.abs(alpha[i]) > m) m = Math.abs(alpha[i]);
        }
        return m;
    }

    private static double[] perTokenAlpha(KsqModel model, int tokenId) {
        double[] logits = model.embedding().lookup(tokenId);
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) out[i] = Math.tanh(logits[i]);
        return out;
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

    private static int argmaxAbs(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[best])) best = i;
        return best;
    }
}
