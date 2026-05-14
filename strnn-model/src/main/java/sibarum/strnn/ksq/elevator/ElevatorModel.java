package sibarum.strnn.ksq.elevator;

import sibarum.strnn.ksq.Mat2;

/**
 * Elevator KSQ — final iter-6 architecture (Phase 2 α-formation rule
 * plus Phase 3's extended anchor set). Unifies direction and magnitude:
 * α = sumLogits directly (no normalization, no tanh). Magnitude flows
 * into Q through the bilinear step rather than being routed around it
 * to the head. With the 5th anchor K_eMinus included
 * ({@link ElevatorAnchors}), the architecture stabilizes at LR=0.1
 * where the no-K_eMinus version of Phase 2 diverged (7/10 NaN).
 *
 * <p>Forward pipeline:
 * <pre>
 *   sumLogits = Σ_t E[token_t]
 *   α         = sumLogits                      (no normalization)
 *   Q         = Σ_i α_i K_i                    (5 anchors, K_eMinus included)
 *   Q²        = Q · Q                          (the one bilinear step)
 *   β_i       = ⟨Q², K_i⟩_F / ‖K_i‖_F²
 *   logits    = W · β + b
 * </pre>
 *
 * <p>Solves T=2 XOR at 10/10 across (λ, ν) at LR=0.1. Falsified at T=4
 * and T=8 by {@link sibarum.strnn.demo.ElevatorMagnitudeClusterDemo}:
 * the single bilinear step Q² is capped at polynomial degree 2 in ℓ,
 * regardless of ‖ℓ‖. See iter-6 falsification in
 * {@code docs/16-ksq-substrate.md}.
 *
 * <p>Per-vocab and cross-vocab regularizers operate on the
 * direction-normalized embedding row, not raw α. This separates
 * "which anchor" specialization from "what level" magnitude;
 * magnitude is unregularized.
 */
public final class ElevatorModel {

    public static final int N = ElevatorAnchors.N;
    public static final double R_EPS = 1e-12;
    public static final double SCALAR_OTHER_THRESHOLD = 0.5;

    private final ElevatorEmbeddingTable embedding;
    private final ElevatorOutputHead head;

    private int[] cachedTokens;
    private double[] cachedSumLogits;
    private double[] cachedAlpha;
    private double[][] cachedQ;
    private double[][] cachedQ2;
    private double[] cachedBeta;
    private double[] cachedLogits;

    public ElevatorModel(int vocab, int outDim, long seed) {
        this(vocab, outDim, seed, 1.0);
    }

    public ElevatorModel(int vocab, int outDim, long seed, double embedInitBound) {
        this.embedding = new ElevatorEmbeddingTable(vocab, N, seed, embedInitBound);
        this.head = new ElevatorOutputHead(outDim, N, seed ^ 0x9E3779B97F4A7C15L);
    }

    public ElevatorEmbeddingTable embedding() { return embedding; }
    public ElevatorOutputHead head() { return head; }
    public int outDim() { return head.outDim(); }
    public int vocab() { return embedding.vocab(); }

    public double[] alpha() {
        if (cachedAlpha == null) throw new IllegalStateException("forward must be called first");
        return cachedAlpha;
    }

    public double magnitude() {
        if (cachedSumLogits == null) throw new IllegalStateException("forward must be called first");
        double s = 0.0;
        for (double v : cachedSumLogits) s += v * v;
        return Math.sqrt(s);
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
            for (int a = 0; a < N; a++) {
                cachedSumLogits[a] += tokLogits[a];
            }
        }

        cachedAlpha = cachedSumLogits.clone();
        cachedQ = anchorCombination(cachedAlpha);
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

        // α = sumLogits identity → dα = dSumLogits.
        double[] dSumLogits = new double[N];
        for (int i = 0; i < N; i++) {
            dSumLogits[i] = Mat2.frobInner(dQ, ElevatorAnchors.matrix(i));
        }

        for (int t = 0; t < cachedTokens.length; t++) {
            embedding.accumulateGradient(cachedTokens[t], dSumLogits);
        }
    }

    /**
     * Per-vocab subalgebra-specialization regularizer, applied to the
     * direction-normalized embedding row α_v = ℓ_v / ||ℓ_v||:
     * λ · Σ_v Σ_{i≠j} α_v[i]² α_v[j]² = λ · Σ_v (1 - Σ_i α_v[i]⁴).
     *
     * <p>Operates on direction only; magnitude is unregularized
     * (per the elevator plan's "separate which-anchor from what-level").
     */
    public double regularizerLoss(double lambda) {
        if (lambda == 0.0) return 0.0;
        double r = 0.0;
        for (int v = 0; v < embedding.vocab(); v++) {
            double[] alpha = normalize(embedding.lookup(v));
            double sumQuad = 0.0;
            for (int i = 0; i < N; i++) {
                double a2 = alpha[i] * alpha[i];
                sumQuad += a2 * a2;
            }
            r += 1.0 - sumQuad;
        }
        return lambda * r;
    }

    public void regularizerBackward(double lambda) {
        if (lambda == 0.0) return;
        for (int v = 0; v < embedding.vocab(); v++) {
            double[] ell = embedding.lookup(v);
            double m = magnitudeOf(ell);
            double[] alpha = new double[N];
            for (int i = 0; i < N; i++) alpha[i] = ell[i] / m;

            double[] dAlpha = new double[N];
            for (int k = 0; k < N; k++) {
                dAlpha[k] = -4.0 * lambda * alpha[k] * alpha[k] * alpha[k];
            }
            embedding.accumulateGradient(v, unitNormBackward(alpha, m, dAlpha));
        }
    }

    /**
     * Cross-vocab contrastive regularizer on direction-normalized rows:
     * ν · Σ_{v1≠v2} ⟨α(v1), α(v2)⟩², where α(v) = ℓ_v / ||ℓ_v||.
     */
    public double crossVocabRegularizerLoss(double nu) {
        if (nu == 0.0) return 0.0;
        int V = embedding.vocab();
        double[][] alphas = new double[V][];
        for (int v = 0; v < V; v++) alphas[v] = normalize(embedding.lookup(v));
        double r = 0.0;
        for (int v1 = 0; v1 < V; v1++) {
            for (int v2 = 0; v2 < V; v2++) {
                if (v1 == v2) continue;
                double c = 0.0;
                for (int i = 0; i < N; i++) c += alphas[v1][i] * alphas[v2][i];
                r += c * c;
            }
        }
        return nu * r;
    }

    public void crossVocabRegularizerBackward(double nu) {
        if (nu == 0.0) return;
        int V = embedding.vocab();
        double[][] alphas = new double[V][];
        double[] mags = new double[V];
        for (int v = 0; v < V; v++) {
            double[] ell = embedding.lookup(v);
            mags[v] = magnitudeOf(ell);
            alphas[v] = new double[N];
            for (int i = 0; i < N; i++) alphas[v][i] = ell[i] / mags[v];
        }

        double[][] c = new double[V][V];
        for (int v1 = 0; v1 < V; v1++) {
            for (int v2 = 0; v2 < V; v2++) {
                if (v1 == v2) continue;
                double s = 0.0;
                for (int i = 0; i < N; i++) s += alphas[v1][i] * alphas[v2][i];
                c[v1][v2] = s;
            }
        }

        for (int va = 0; va < V; va++) {
            double[] dAlpha = new double[N];
            for (int vb = 0; vb < V; vb++) {
                if (va == vb) continue;
                double scale = 4.0 * nu * c[va][vb];
                for (int k = 0; k < N; k++) dAlpha[k] += scale * alphas[vb][k];
            }
            embedding.accumulateGradient(va, unitNormBackward(alphas[va], mags[va], dAlpha));
        }
    }

    public void step(double lr) {
        embedding.step(lr);
        head.step(lr);
    }

    /**
     * SGD step with global gradient-norm clipping. Safety measure for
     * the unbounded-magnitude regime: computes the total L2 norm of
     * all gradients (embedding + head), and if it exceeds {@code
     * clipNorm}, scales all gradients by {@code clipNorm / totalNorm}
     * before applying SGD. Per the elevator plan's risk register,
     * this is *not* a magnitude cap on parameters — it's a per-step
     * safety on gradient size, preserving direction.
     */
    public void stepClipped(double lr, double clipNorm) {
        double sumSq = 0.0;
        for (int v = 0; v < embedding.vocab(); v++) {
            double[] g = embedding.gradient(v);
            for (int i = 0; i < N; i++) sumSq += g[i] * g[i];
        }
        for (int k = 0; k < head.outDim(); k++) {
            double[] gw = head.gradWeights()[k];
            for (int i = 0; i < head.inputDim(); i++) sumSq += gw[i] * gw[i];
            double gb = head.gradBiases()[k];
            sumSq += gb * gb;
        }
        double norm = Math.sqrt(sumSq);
        if (norm > clipNorm) {
            double scale = clipNorm / norm;
            for (int v = 0; v < embedding.vocab(); v++) {
                double[] g = embedding.gradient(v);
                for (int i = 0; i < N; i++) g[i] *= scale;
            }
            for (int k = 0; k < head.outDim(); k++) {
                double[] gw = head.gradWeights()[k];
                for (int i = 0; i < head.inputDim(); i++) gw[i] *= scale;
                head.gradBiases()[k] *= scale;
            }
        }
        embedding.step(lr);
        head.step(lr);
    }

    public void zeroGrad() {
        embedding.zeroGrad();
        head.zeroGrad();
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

    /**
     * Backward through α = ℓ/||ℓ||. Given dL/dα (dAlpha) and the current
     * α (already normalized) plus magnitude m = ||ℓ||, returns dL/dℓ:
     * <pre>
     *   dL/dℓ_j = (dAlpha_j - α_j · ⟨α, dAlpha⟩) / m
     * </pre>
     * Standard L2-normalize Jacobian projection.
     */
    private static double[] unitNormBackward(double[] alpha, double m, double[] dAlpha) {
        double dot = 0.0;
        for (int i = 0; i < alpha.length; i++) dot += alpha[i] * dAlpha[i];
        double[] dL = new double[alpha.length];
        for (int j = 0; j < alpha.length; j++) {
            dL[j] = (dAlpha[j] - alpha[j] * dot) / m;
        }
        return dL;
    }

    private static double[] normalize(double[] ell) {
        double m = magnitudeOf(ell);
        double[] out = new double[ell.length];
        for (int i = 0; i < ell.length; i++) out[i] = ell[i] / m;
        return out;
    }

    private static double magnitudeOf(double[] ell) {
        double s = 0.0;
        for (double v : ell) s += v * v;
        return Math.sqrt(s + R_EPS);
    }

    private static double[][] anchorCombination(double[] alpha) {
        double[][] q = new double[2][2];
        for (int i = 0; i < ElevatorAnchors.N; i++) {
            Mat2.addScaledInPlace(q, ElevatorAnchors.matrix(i), alpha[i]);
        }
        return q;
    }

    private static double[] anchorProjection(double[][] s) {
        double[] beta = new double[ElevatorAnchors.N];
        for (int i = 0; i < ElevatorAnchors.N; i++) {
            beta[i] = Mat2.frobInner(s, ElevatorAnchors.matrix(i)) / ElevatorAnchors.frobNormSq(i);
        }
        return beta;
    }
}
