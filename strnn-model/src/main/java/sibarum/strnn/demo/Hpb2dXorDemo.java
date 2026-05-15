package sibarum.strnn.demo;

import sibarum.strnn.hpb.HarmonicBasis;
import sibarum.strnn.hpb.PiecewisePolynomial;

import java.util.Random;

/**
 * 2D XOR over independent binary inputs. Three architectures, each
 * stressing a different aspect of how the harmonic basis composes
 * across input dimensions:
 *
 * <ol>
 *   <li><b>Per-dim lift + linear readout (predicted: FAIL).</b>
 *       Lifting is linear in each input; a linear readout cannot
 *       multiply features across dimensions. Confirms that iter-1's
 *       T5 XOR success depended on a 1D encoding (four corners along
 *       a single period), not on the basis being multi-dim capable.</li>
 *   <li><b>Joint tensor-product lift + linear readout (predicted:
 *       SOLVE).</b> Features are outer products of per-dim lifts:
 *       featDim = (2K)². The cross-term {@code sq_1(x_0) · sq_1(x_1)}
 *       is exactly the XOR sign pattern. K=1 (featDim=4) suffices.</li>
 *   <li><b>Per-dim lift + hidden ReLU + linear readout (predicted:
 *       SOLVE).</b> Standard 2-layer MLP on top of per-dim features;
 *       the hidden non-linearity recovers the cross-dim multiplication
 *       the linear readout cannot. Baseline confirmation that HPB
 *       plugs into the usual MLP architecture as a feature lift.</li>
 * </ol>
 *
 * <p>Inputs: each bit b ∈ {0, 1} mapped to scalar x ∈ {0.125, 0.625}
 * (mid-piece for k=1 and k=2). Standard XOR labels.
 */
public final class Hpb2dXorDemo {

    private static final int OUT_DIM = 2;
    private static final int EPOCHS = 5000;
    private static final double LR = 0.05;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L };

    public static void main(String[] args) {
        double[][] xs = {
                { 0.125, 0.125 },
                { 0.125, 0.625 },
                { 0.625, 0.125 },
                { 0.625, 0.625 }
        };
        int[] ys = { 0, 1, 1, 0 };

        System.out.println();
        System.out.println("=== HPB 2D XOR — three architectures ===");
        System.out.println("  inputs: each bit b mapped to x ∈ {0.125, 0.625}");
        System.out.println("  epochs=" + EPOCHS + "  lr=" + LR + "  seeds=" + SEEDS.length);

        System.out.println();
        System.out.println("--- A. Per-dim lift + linear readout (predicted: FAIL) ---");
        for (int K : new int[] { 1, 2, 4 }) {
            int solved = 0;
            double bestCe = Double.POSITIVE_INFINITY;
            for (long seed : SEEDS) {
                PerDimModel m = new PerDimModel(K, OUT_DIM, seed);
                for (int e = 0; e < EPOCHS; e++) {
                    m.zeroGrad();
                    for (int i = 0; i < xs.length; i++) { m.forward(xs[i]); m.backward(ys[i]); }
                    m.step(LR / xs.length);
                }
                if (m.allCorrect(xs, ys)) solved++;
                bestCe = Math.min(bestCe, m.epochLoss(xs, ys));
            }
            System.out.printf("  K=%d (featDim=%d): %d/%d solved   best CE=%.4e%n",
                    K, 4 * K, solved, SEEDS.length, bestCe);
        }

        System.out.println();
        System.out.println("--- B. Joint tensor-product lift + linear readout (predicted: SOLVE) ---");
        for (int K : new int[] { 1, 2 }) {
            int solved = 0;
            double bestCe = Double.POSITIVE_INFINITY;
            for (long seed : SEEDS) {
                JointModel m = new JointModel(K, OUT_DIM, seed);
                for (int e = 0; e < EPOCHS; e++) {
                    m.zeroGrad();
                    for (int i = 0; i < xs.length; i++) { m.forward(xs[i]); m.backward(ys[i]); }
                    m.step(LR / xs.length);
                }
                if (m.allCorrect(xs, ys)) solved++;
                bestCe = Math.min(bestCe, m.epochLoss(xs, ys));
            }
            int feat = (2 * K) * (2 * K);
            System.out.printf("  K=%d (featDim=%d): %d/%d solved   best CE=%.4e%n",
                    K, feat, solved, SEEDS.length, bestCe);
        }

        System.out.println();
        System.out.println("--- C. Per-dim lift + hidden ReLU + linear readout (predicted: SOLVE) ---");
        for (int K : new int[] { 1, 2 }) {
            for (int H : new int[] { 4, 8 }) {
                int solved = 0;
                double bestCe = Double.POSITIVE_INFINITY;
                for (long seed : SEEDS) {
                    HiddenModel m = new HiddenModel(K, H, OUT_DIM, seed);
                    for (int e = 0; e < EPOCHS; e++) {
                        m.zeroGrad();
                        for (int i = 0; i < xs.length; i++) { m.forward(xs[i]); m.backward(ys[i]); }
                        m.step(LR / xs.length);
                    }
                    if (m.allCorrect(xs, ys)) solved++;
                    bestCe = Math.min(bestCe, m.epochLoss(xs, ys));
                }
                System.out.printf("  K=%d hidden=%d (featDim=%d): %d/%d solved   best CE=%.4e%n",
                        K, H, 4 * K, solved, SEEDS.length, bestCe);
            }
        }
    }

    private static double[] liftScalar(double x, PiecewisePolynomial[] tri, PiecewisePolynomial[] sq) {
        int K = tri.length;
        double[] f = new double[2 * K];
        for (int i = 0; i < K; i++) {
            f[2 * i]     = tri[i].evaluate(x);
            f[2 * i + 1] = sq[i].evaluate(x);
        }
        return f;
    }

    private static double[] softmax(double[] logits) {
        int n = logits.length;
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logits) if (v > m) m = v;
        double[] out = new double[n];
        double s = 0.0;
        for (int i = 0; i < n; i++) { out[i] = Math.exp(logits[i] - m); s += out[i]; }
        for (int i = 0; i < n; i++) out[i] /= s;
        return out;
    }

    private static double ce(double[] logits, int target) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logits) if (v > m) m = v;
        double s = 0.0;
        for (double v : logits) s += Math.exp(v - m);
        return Math.log(s) + m - logits[target];
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    static final class PerDimModel {
        final int K, outDim, featDim;
        final PiecewisePolynomial[] tri, sq;
        final double[][] W;
        final double[] b;
        final double[][] gW;
        final double[] gB;
        double[] cachedFeats;
        double[] cachedLogits;

        PerDimModel(int K, int outDim, long seed) {
            this.K = K;
            this.outDim = outDim;
            this.featDim = 4 * K;
            this.tri = new PiecewisePolynomial[K];
            this.sq  = new PiecewisePolynomial[K];
            for (int i = 0; i < K; i++) { tri[i] = HarmonicBasis.triK(i + 1); sq[i] = HarmonicBasis.sqK(i + 1); }
            this.W = new double[outDim][featDim];
            this.b = new double[outDim];
            this.gW = new double[outDim][featDim];
            this.gB = new double[outDim];
            Random rng = new Random(seed);
            double bnd = Math.sqrt(6.0 / (featDim + outDim));
            for (int o = 0; o < outDim; o++)
                for (int f = 0; f < featDim; f++)
                    W[o][f] = (rng.nextDouble() * 2.0 - 1.0) * bnd;
        }

        double[] forward(double[] x) {
            double[] f0 = liftScalar(x[0], tri, sq);
            double[] f1 = liftScalar(x[1], tri, sq);
            cachedFeats = new double[featDim];
            System.arraycopy(f0, 0, cachedFeats, 0, 2 * K);
            System.arraycopy(f1, 0, cachedFeats, 2 * K, 2 * K);
            cachedLogits = new double[outDim];
            for (int o = 0; o < outDim; o++) {
                double s = b[o];
                for (int f = 0; f < featDim; f++) s += W[o][f] * cachedFeats[f];
                cachedLogits[o] = s;
            }
            return cachedLogits;
        }

        void backward(int target) {
            double[] dL = softmax(cachedLogits);
            dL[target] -= 1.0;
            for (int o = 0; o < outDim; o++) {
                gB[o] += dL[o];
                for (int f = 0; f < featDim; f++) gW[o][f] += dL[o] * cachedFeats[f];
            }
        }

        void step(double lr) {
            for (int o = 0; o < outDim; o++) {
                b[o] -= lr * gB[o]; gB[o] = 0.0;
                for (int f = 0; f < featDim; f++) { W[o][f] -= lr * gW[o][f]; gW[o][f] = 0.0; }
            }
        }

        void zeroGrad() {
            for (int o = 0; o < outDim; o++) {
                gB[o] = 0.0;
                for (int f = 0; f < featDim; f++) gW[o][f] = 0.0;
            }
        }

        boolean allCorrect(double[][] xs, int[] ys) {
            for (int i = 0; i < xs.length; i++) if (argmax(forward(xs[i])) != ys[i]) return false;
            return true;
        }

        double epochLoss(double[][] xs, int[] ys) {
            double s = 0.0;
            for (int i = 0; i < xs.length; i++) { forward(xs[i]); s += ce(cachedLogits, ys[i]); }
            return s / xs.length;
        }
    }

    static final class JointModel {
        final int K, outDim, featDim, perDim;
        final PiecewisePolynomial[] tri, sq;
        final double[][] W;
        final double[] b;
        final double[][] gW;
        final double[] gB;
        double[] cachedFeats;
        double[] cachedLogits;

        JointModel(int K, int outDim, long seed) {
            this.K = K;
            this.outDim = outDim;
            this.perDim = 2 * K;
            this.featDim = perDim * perDim;
            this.tri = new PiecewisePolynomial[K];
            this.sq  = new PiecewisePolynomial[K];
            for (int i = 0; i < K; i++) { tri[i] = HarmonicBasis.triK(i + 1); sq[i] = HarmonicBasis.sqK(i + 1); }
            this.W = new double[outDim][featDim];
            this.b = new double[outDim];
            this.gW = new double[outDim][featDim];
            this.gB = new double[outDim];
            Random rng = new Random(seed);
            double bnd = Math.sqrt(6.0 / (featDim + outDim));
            for (int o = 0; o < outDim; o++)
                for (int f = 0; f < featDim; f++)
                    W[o][f] = (rng.nextDouble() * 2.0 - 1.0) * bnd;
        }

        double[] forward(double[] x) {
            double[] f0 = liftScalar(x[0], tri, sq);
            double[] f1 = liftScalar(x[1], tri, sq);
            cachedFeats = new double[featDim];
            for (int i = 0; i < perDim; i++)
                for (int j = 0; j < perDim; j++)
                    cachedFeats[i * perDim + j] = f0[i] * f1[j];
            cachedLogits = new double[outDim];
            for (int o = 0; o < outDim; o++) {
                double s = b[o];
                for (int f = 0; f < featDim; f++) s += W[o][f] * cachedFeats[f];
                cachedLogits[o] = s;
            }
            return cachedLogits;
        }

        void backward(int target) {
            double[] dL = softmax(cachedLogits);
            dL[target] -= 1.0;
            for (int o = 0; o < outDim; o++) {
                gB[o] += dL[o];
                for (int f = 0; f < featDim; f++) gW[o][f] += dL[o] * cachedFeats[f];
            }
        }

        void step(double lr) {
            for (int o = 0; o < outDim; o++) {
                b[o] -= lr * gB[o]; gB[o] = 0.0;
                for (int f = 0; f < featDim; f++) { W[o][f] -= lr * gW[o][f]; gW[o][f] = 0.0; }
            }
        }

        void zeroGrad() {
            for (int o = 0; o < outDim; o++) {
                gB[o] = 0.0;
                for (int f = 0; f < featDim; f++) gW[o][f] = 0.0;
            }
        }

        boolean allCorrect(double[][] xs, int[] ys) {
            for (int i = 0; i < xs.length; i++) if (argmax(forward(xs[i])) != ys[i]) return false;
            return true;
        }

        double epochLoss(double[][] xs, int[] ys) {
            double s = 0.0;
            for (int i = 0; i < xs.length; i++) { forward(xs[i]); s += ce(cachedLogits, ys[i]); }
            return s / xs.length;
        }
    }

    static final class HiddenModel {
        final int K, hidden, outDim, perDim, featDim;
        final PiecewisePolynomial[] tri, sq;
        final double[][] W1;
        final double[] b1;
        final double[][] gW1;
        final double[] gB1;
        final double[][] W2;
        final double[] b2;
        final double[][] gW2;
        final double[] gB2;
        double[] cachedFeats;
        double[] cachedZ1;
        double[] cachedA1;
        double[] cachedLogits;

        HiddenModel(int K, int hidden, int outDim, long seed) {
            this.K = K;
            this.hidden = hidden;
            this.outDim = outDim;
            this.perDim = 2 * K;
            this.featDim = 4 * K;
            this.tri = new PiecewisePolynomial[K];
            this.sq  = new PiecewisePolynomial[K];
            for (int i = 0; i < K; i++) { tri[i] = HarmonicBasis.triK(i + 1); sq[i] = HarmonicBasis.sqK(i + 1); }
            this.W1 = new double[hidden][featDim];
            this.b1 = new double[hidden];
            this.gW1 = new double[hidden][featDim];
            this.gB1 = new double[hidden];
            this.W2 = new double[outDim][hidden];
            this.b2 = new double[outDim];
            this.gW2 = new double[outDim][hidden];
            this.gB2 = new double[outDim];
            Random rng = new Random(seed);
            double bnd1 = Math.sqrt(6.0 / (featDim + hidden));
            for (int h = 0; h < hidden; h++)
                for (int f = 0; f < featDim; f++)
                    W1[h][f] = (rng.nextDouble() * 2.0 - 1.0) * bnd1;
            double bnd2 = Math.sqrt(6.0 / (hidden + outDim));
            for (int o = 0; o < outDim; o++)
                for (int h = 0; h < hidden; h++)
                    W2[o][h] = (rng.nextDouble() * 2.0 - 1.0) * bnd2;
        }

        double[] forward(double[] x) {
            double[] f0 = liftScalar(x[0], tri, sq);
            double[] f1 = liftScalar(x[1], tri, sq);
            cachedFeats = new double[featDim];
            System.arraycopy(f0, 0, cachedFeats, 0, perDim);
            System.arraycopy(f1, 0, cachedFeats, perDim, perDim);
            cachedZ1 = new double[hidden];
            cachedA1 = new double[hidden];
            for (int h = 0; h < hidden; h++) {
                double s = b1[h];
                for (int f = 0; f < featDim; f++) s += W1[h][f] * cachedFeats[f];
                cachedZ1[h] = s;
                cachedA1[h] = s > 0 ? s : 0;
            }
            cachedLogits = new double[outDim];
            for (int o = 0; o < outDim; o++) {
                double s = b2[o];
                for (int h = 0; h < hidden; h++) s += W2[o][h] * cachedA1[h];
                cachedLogits[o] = s;
            }
            return cachedLogits;
        }

        void backward(int target) {
            double[] dLogits = softmax(cachedLogits);
            dLogits[target] -= 1.0;
            double[] dA1 = new double[hidden];
            for (int o = 0; o < outDim; o++) {
                gB2[o] += dLogits[o];
                for (int h = 0; h < hidden; h++) {
                    gW2[o][h] += dLogits[o] * cachedA1[h];
                    dA1[h] += W2[o][h] * dLogits[o];
                }
            }
            double[] dZ1 = new double[hidden];
            for (int h = 0; h < hidden; h++) dZ1[h] = cachedZ1[h] > 0 ? dA1[h] : 0;
            for (int h = 0; h < hidden; h++) {
                gB1[h] += dZ1[h];
                for (int f = 0; f < featDim; f++) gW1[h][f] += dZ1[h] * cachedFeats[f];
            }
        }

        void step(double lr) {
            for (int o = 0; o < outDim; o++) {
                b2[o] -= lr * gB2[o]; gB2[o] = 0.0;
                for (int h = 0; h < hidden; h++) { W2[o][h] -= lr * gW2[o][h]; gW2[o][h] = 0.0; }
            }
            for (int h = 0; h < hidden; h++) {
                b1[h] -= lr * gB1[h]; gB1[h] = 0.0;
                for (int f = 0; f < featDim; f++) { W1[h][f] -= lr * gW1[h][f]; gW1[h][f] = 0.0; }
            }
        }

        void zeroGrad() {
            for (int o = 0; o < outDim; o++) {
                gB2[o] = 0.0;
                for (int h = 0; h < hidden; h++) gW2[o][h] = 0.0;
            }
            for (int h = 0; h < hidden; h++) {
                gB1[h] = 0.0;
                for (int f = 0; f < featDim; f++) gW1[h][f] = 0.0;
            }
        }

        boolean allCorrect(double[][] xs, int[] ys) {
            for (int i = 0; i < xs.length; i++) if (argmax(forward(xs[i])) != ys[i]) return false;
            return true;
        }

        double epochLoss(double[][] xs, int[] ys) {
            double s = 0.0;
            for (int i = 0; i < xs.length; i++) { forward(xs[i]); s += ce(cachedLogits, ys[i]); }
            return s / xs.length;
        }
    }
}
