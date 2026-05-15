package sibarum.strnn.demo;

import sibarum.strnn.hpb.HpbModel;

/**
 * HPB parity sweep: T ∈ {2, 4, 8}, K ∈ {1, 2, 4, 8, 16}. For each T,
 * encode the 2^T input patterns as scalars x_n = (n + 0.5) / 2^T,
 * label by {@code popcount(n) mod 2}. The grid finds the smallest K
 * that lets a linear readout over the harmonic basis solve T-bit
 * parity.
 *
 * <p>Why this matters: iter-1's T5 XOR demo showed K=1 + linear solves
 * XOR — but only because at N=4 the parity sign pattern coincides
 * with sq_1 evaluated at the four chosen positions. A square wave
 * naturally produces parity within one period. The real capacity
 * question is whether the basis solves parity at higher T, where the
 * label pattern is more structured and may require multiple
 * frequencies in the lift.
 *
 * <p>Direct cross-architecture comparison: KSQ's iter-6 falsified at
 * T=4 and T=8 parity (the bilinear step Q² has a degree-2 expressivity
 * ceiling regardless of magnitude). HPB has a different expressivity
 * profile — sq_k contributes all odd harmonics of k by Fourier
 * series; the question is whether that's enough to express parity at
 * each T.
 */
public final class HpbParityDemo {

    private static final int OUT_DIM = 2;
    private static final int[] TS = { 2, 4, 8 };
    private static final int[] KS = { 1, 2, 4, 8, 16 };
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L };
    private static final int EPOCHS = 5000;
    private static final double LR = 0.05;

    public static void main(String[] args) {
        System.out.println();
        System.out.println("=== HPB parity sweep (T6) ===");
        System.out.println("  T ∈ " + java.util.Arrays.toString(TS)
                + "   K ∈ " + java.util.Arrays.toString(KS)
                + "   seeds=" + SEEDS.length
                + "   epochs=" + EPOCHS + "   lr=" + LR);
        System.out.println();
        System.out.printf("  %-6s", "T \\ K");
        for (int K : KS) System.out.printf("  K=%-3d", K);
        System.out.println("    (solve rate out of " + SEEDS.length + ")");
        System.out.print("  ------");
        for (int K : KS) System.out.print("-------");
        System.out.println();

        for (int T : TS) {
            int N = 1 << T;
            double[] xs = new double[N];
            int[] ys = new int[N];
            for (int n = 0; n < N; n++) {
                xs[n] = (n + 0.5) / N;
                ys[n] = Integer.bitCount(n) & 1;
            }

            System.out.printf("   T=%-3d", T);
            for (int K : KS) {
                int solved = 0;
                for (long seed : SEEDS) {
                    HpbModel model = new HpbModel(K, OUT_DIM, seed);
                    for (int epoch = 0; epoch < EPOCHS; epoch++) {
                        model.zeroGrad();
                        for (int i = 0; i < N; i++) {
                            model.forward(xs[i]);
                            model.backward(ys[i]);
                        }
                        model.step(LR / N);
                    }
                    if (allCorrect(model, xs, ys)) solved++;
                }
                System.out.printf("  %d/%-3d ", solved, SEEDS.length);
            }
            System.out.println();
        }
    }

    private static boolean allCorrect(HpbModel m, double[] xs, int[] ys) {
        for (int i = 0; i < xs.length; i++) {
            double[] logits = m.forward(xs[i]);
            if (argmax(logits) != ys[i]) return false;
        }
        return true;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
