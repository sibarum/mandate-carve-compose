package sibarum.strnn.ksq.elevator;

import sibarum.strnn.ksq.Mat2;

/**
 * Iter 7 — signed-power level activation. The architecture adds one new
 * scalar parameter $n$ (the level) and one new op between the pooled
 * embedding and the algebra lift:
 *
 * <pre>
 *   sumLogits = Σ_t E[token_t]                   (sum-pool, as iter-6 Phase 2)
 *   y_i = sign(sumLogits_i) · |sumLogits_i|^n    (per-component signed power)
 *   Q = Σ_i y_i K_i                              (lift, unchanged from iter 5/6)
 *   S = Q · Q                                    (single bilinear step)
 *   β_i = ⟨S, K_i⟩_F / ‖K_i‖_F²                  (readout)
 *   logits = W · β + b                           (head)
 * </pre>
 *
 * <p>Polynomial degree of output in sumLogits is $2n$. Continuous in $n$.
 * Initialize $n = 1$ to match iter-6 Phase 2 exactly; training moves $n$
 * up or down based on what polynomial degree the task wants.
 *
 * <p>$n$ is unconstrained. No softplus reparam, no clamping. If $n$ drifts
 * below 1 the gradient $\partial y / \partial \ell = n |\ell|^{n-1}$
 * diverges as $|\ell| \to 0$; that's a real algebraic fact about this
 * activation, not a defect to engineer around. The MCC methodology says
 * to observe the dynamics, not suppress them. (See feedback memory
 * "no normalization crutches.")
 *
 * <p>At $s_i = 0$ exactly, the values used are the continuous limits:
 * $y = 0$, $\partial y / \partial n = 0$. $\partial y / \partial \ell$
 * follows the pointwise $n |\ell|^{n-1}$ formula (= 0 for $n > 1$, = 1 for
 * $n = 1$, $+\infty$ for $n < 1$ — Java's `Math.pow(0, k)` handles each).
 */
public final class PowerLevelModel {

    public static final int N = ElevatorAnchors.N;

    private final ElevatorEmbeddingTable embedding;
    private final ElevatorOutputHead head;
    private double n;
    private double gradN;

    private int[] cachedTokens;
    private double[] cachedSumLogits;
    private double[] cachedY;
    private double[][] cachedQ;
    private double[][] cachedQ2;
    private double[] cachedBeta;
    private double[] cachedLogits;

    public PowerLevelModel(int vocab, int outDim, long seed) {
        this(vocab, outDim, seed, 1.0, 1.0);
    }

    public PowerLevelModel(int vocab, int outDim, long seed,
                           double embedInitBound, double nInit) {
        this.embedding = new ElevatorEmbeddingTable(vocab, N, seed, embedInitBound);
        this.head = new ElevatorOutputHead(outDim, N, seed ^ 0x9E3779B97F4A7C15L);
        this.n = nInit;
        this.gradN = 0.0;
    }

    public ElevatorEmbeddingTable embedding() { return embedding; }
    public ElevatorOutputHead head() { return head; }
    public int outDim() { return head.outDim(); }
    public int vocab() { return embedding.vocab(); }
    public double n() { return n; }
    public double gradN() { return gradN; }

    public void setN(double newN) { this.n = newN; }

    public double[] alpha() {
        if (cachedY == null) throw new IllegalStateException("forward must be called first");
        return cachedY;
    }

    public double[][] q() {
        if (cachedQ == null) throw new IllegalStateException("forward must be called first");
        return cachedQ;
    }

    public double[][] qSquared() {
        if (cachedQ2 == null) throw new IllegalStateException("forward must be called first");
        return cachedQ2;
    }

    public double[] forward(int[] tokens) {
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException("tokens must be non-empty");
        }

        cachedTokens = tokens.clone();
        cachedSumLogits = new double[N];
        for (int t = 0; t < tokens.length; t++) {
            double[] tokLogits = embedding.lookup(tokens[t]);
            for (int a = 0; a < N; a++) cachedSumLogits[a] += tokLogits[a];
        }

        cachedY = new double[N];
        for (int i = 0; i < N; i++) {
            double s = cachedSumLogits[i];
            // y = sign(s) · |s|^n; at s=0 the continuous extension is 0 (for n > 0).
            if (s == 0.0) {
                cachedY[i] = 0.0;
            } else {
                cachedY[i] = Math.signum(s) * Math.pow(Math.abs(s), n);
            }
        }

        cachedQ = anchorCombination(cachedY);
        cachedQ2 = Mat2.mul(cachedQ, cachedQ);
        cachedBeta = anchorProjection(cachedQ2);
        cachedLogits = head.forward(cachedBeta);
        return cachedLogits.clone();
    }

    public double crossEntropyLoss(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");
        if (target < 0 || target >= cachedLogits.length) {
            throw new IllegalArgumentException("target out of range: " + target);
        }
        double m = Double.NEGATIVE_INFINITY;
        for (double v : cachedLogits) if (v > m) m = v;
        double sumExp = 0.0;
        for (double v : cachedLogits) sumExp += Math.exp(v - m);
        return Math.log(sumExp) + m - cachedLogits[target];
    }

    public void backward(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");

        double[] dLogits = softmax(cachedLogits);
        dLogits[target] -= 1.0;

        double[] dBeta = head.backward(dLogits);

        double[][] dQ2 = new double[2][2];
        for (int i = 0; i < N; i++) {
            double scale = dBeta[i] / ElevatorAnchors.frobNormSq(i);
            Mat2.addScaledInPlace(dQ2, ElevatorAnchors.matrix(i), scale);
        }

        double[][] qT = Mat2.transpose(cachedQ);
        double[][] dQfromLeft = Mat2.mul(dQ2, qT);
        double[][] dQfromRight = Mat2.mul(qT, dQ2);
        double[][] dQ = new double[2][2];
        Mat2.addScaledInPlace(dQ, dQfromLeft, 1.0);
        Mat2.addScaledInPlace(dQ, dQfromRight, 1.0);

        double[] dY = new double[N];
        for (int i = 0; i < N; i++) {
            dY[i] = Mat2.frobInner(dQ, ElevatorAnchors.matrix(i));
        }

        // Signed-power backward:
        //   dy/ds = n · |s|^(n-1)     (positive on both sides of 0, by symmetry)
        //   dy/dn = sign(s) · |s|^n · ln|s|
        // At s=0: dy/ds follows the pointwise n·0^(n-1) formula (Java Math.pow
        // returns 0 for k>0, 1 for k=0, +∞ for k<0). dy/dn at s=0 has the
        // 0·(-∞) form pointwise; the well-defined limit is 0, so we use 0.
        double[] dSumLogits = new double[N];
        for (int i = 0; i < N; i++) {
            double s = cachedSumLogits[i];
            if (s == 0.0) {
                // dy/ds = n · 0^(n-1). For n>1: 0. For n=1: 1. For n<1: +∞.
                dSumLogits[i] = dY[i] * n * Math.pow(0.0, n - 1.0);
                // dy/dn contribution at s=0 is 0 by limit; skip.
            } else {
                double absS = Math.abs(s);
                dSumLogits[i] = dY[i] * n * Math.pow(absS, n - 1.0);
                gradN += dY[i] * Math.signum(s) * Math.pow(absS, n) * Math.log(absS);
            }
        }

        for (int t = 0; t < cachedTokens.length; t++) {
            embedding.accumulateGradient(cachedTokens[t], dSumLogits);
        }
    }

    public void step(double lr) {
        embedding.step(lr);
        head.step(lr);
        n -= lr * gradN;
        gradN = 0.0;
    }

    /** Step everything except n (used for ablation: freeze n at its current value). */
    public void stepFrozenN(double lr) {
        embedding.step(lr);
        head.step(lr);
        gradN = 0.0;
    }

    public void zeroGrad() {
        embedding.zeroGrad();
        head.zeroGrad();
        gradN = 0.0;
    }

    private static double[] softmax(double[] logits) {
        int n = logits.length;
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logits) if (v > m) m = v;
        double[] out = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            out[i] = Math.exp(logits[i] - m);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    private static double[][] anchorCombination(double[] y) {
        double[][] q = new double[2][2];
        for (int i = 0; i < N; i++) {
            Mat2.addScaledInPlace(q, ElevatorAnchors.matrix(i), y[i]);
        }
        return q;
    }

    private static double[] anchorProjection(double[][] s) {
        double[] beta = new double[N];
        for (int i = 0; i < N; i++) {
            beta[i] = Mat2.frobInner(s, ElevatorAnchors.matrix(i)) / ElevatorAnchors.frobNormSq(i);
        }
        return beta;
    }
}
