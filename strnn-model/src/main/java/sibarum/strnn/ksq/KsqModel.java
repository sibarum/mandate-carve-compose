package sibarum.strnn.ksq;

/**
 * KSQ end-to-end model. The architecture's distinguishing claim — vs. a
 * transformer — is that the algebra is touched <b>exactly once</b> per
 * forward, in a single bilinear step. Sequence structure is handled at the
 * embedding level by sum-pooling token logits before the softmax; there is
 * no chain of algebra multiplications.
 *
 * <p>Forward pipeline:
 * <pre>
 *   sumLogits = Σ_t E[token_t]                   (sum-pool at the embedding level)
 *   α = tanh(sumLogits)                          (signed anchor coefficients, α ∈ [-1, 1]^n)
 *   Q = Σ_i α_i · K_i                            (lift into the algebra)
 *   Q² = Q · Q                                   (the one bilinear step in M_2(R))
 *   β_i = ⟨Q², K_i⟩_F / ‖K_i‖_F²                (read back as anchor coefficients)
 *   logits = W · β + b                           (linear output head)
 * </pre>
 *
 * <p>Q² is the algebra-flavored non-linearity: it lifts α to quadratic
 * features through the algebra's bilinear product. Without it the whole
 * pipeline collapses to two linear maps; with it, tasks like XOR — and more
 * generally any quadratic-in-pooled-α decision rule — become expressible.
 *
 * <p>α is unconstrained-in-sign (tanh, not softmax). Cross-terms in Q² like
 * β_i = 2 α_0 α_i are sign-controlled, so elliptic specialization (K_i,
 * K_i² = -I) is on equal footing with hyperbolic specialization (K_j,
 * K_j² = +I): both can produce non-trivial signed cross-terms in the
 * readout. With softmax α, β_i could only take one sign, which structurally
 * privileged K_j over K_i in the optimizer's basin.
 *
 * <p>This is the prototype shortcut Q = Σ α_i K_i (linear combination, not
 * the geodesic Q = exp(Σ α_i log K_i)). Differentiable everywhere; verified
 * by finite-difference gradient check in {@code KsqGradientCheckDemo}.
 */
public final class KsqModel {

    private final KsqEmbeddingTable embedding;
    private final KsqOutputHead head;

    private int[] cachedTokens;
    private double[] cachedSumLogits;
    private double[] cachedAlpha;
    private double[][] cachedQ;
    private double[][] cachedQ2;
    private double[] cachedBeta;
    private double[] cachedLogits;

    public KsqModel(int vocab, int outDim, long seed) {
        this(vocab, outDim, seed, 1.0);
    }

    public KsqModel(int vocab, int outDim, long seed, double embedInitBound) {
        this.embedding = new KsqEmbeddingTable(vocab, KsqAnchors.N, seed, embedInitBound);
        this.head = new KsqOutputHead(outDim, KsqAnchors.N, seed ^ 0x9E3779B97F4A7C15L);
    }

    public KsqEmbeddingTable embedding() { return embedding; }
    public KsqOutputHead head() { return head; }
    public int outDim() { return head.outDim(); }
    public int vocab() { return embedding.vocab(); }

    /** Returns the single pooled α from the most recent forward pass. */
    public double[] alpha() {
        if (cachedAlpha == null) throw new IllegalStateException("forward must be called first");
        return cachedAlpha;
    }

    /** Returns the algebra element Q (before squaring) from the most recent forward. */
    public double[][] q() {
        if (cachedQ == null) throw new IllegalStateException("forward must be called first");
        return cachedQ;
    }

    /** Returns Q² (after the bilinear step) from the most recent forward. */
    public double[][] qSquared() {
        if (cachedQ2 == null) throw new IllegalStateException("forward must be called first");
        return cachedQ2;
    }

    /** Runs the full forward and returns the output logits. */
    public double[] forward(int[] tokens) {
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException("tokens must be non-empty");
        }
        int n = KsqAnchors.N;

        cachedTokens = tokens.clone();
        cachedSumLogits = new double[n];
        for (int t = 0; t < tokens.length; t++) {
            double[] tokLogits = embedding.lookup(tokens[t]);
            for (int a = 0; a < n; a++) {
                cachedSumLogits[a] += tokLogits[a];
            }
        }

        cachedAlpha = tanh(cachedSumLogits);
        cachedQ = anchorCombination(cachedAlpha);
        cachedQ2 = Mat2.mul(cachedQ, cachedQ);
        cachedBeta = anchorProjection(cachedQ2);
        cachedLogits = head.forward(cachedBeta);
        return cachedLogits.clone();
    }

    /**
     * Cross-entropy loss against {@code target} on the most recent forward.
     * Numerically stable via log-sum-exp.
     */
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

    /**
     * Backward pass under softmax-cross-entropy. Assumes forward() was just called.
     * Accumulates gradients into the embedding table and the output head.
     *
     * <p>Backward flow (in reverse of forward):
     * <pre>
     *   dLogits = softmax(logits) - oneHot(target)
     *   dβ ← head.backward(dLogits)
     *   dQ² = Σ_i dβ_i · K_i / ‖K_i‖²
     *   dQ = dQ² · Q^T + Q^T · dQ²             (bilinear step: S=A·B|_{A=B=Q})
     *   dα_i = ⟨dQ, K_i⟩_F
     *   dSumLogits = α ⊙ (dα − ⟨dα, α⟩)        (softmax Jacobian-vector)
     *   for each token t: emb.gradient[token_t] += dSumLogits
     * </pre>
     * Repeated tokens accumulate the same dSumLogits multiple times — correct,
     * because sumLogits is linear in each token's lookup and the count
     * appears as the multiplicity of the addition.
     */
    public void backward(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");
        int n = KsqAnchors.N;

        double[] dLogits = softmax(cachedLogits);
        dLogits[target] -= 1.0;

        double[] dBeta = head.backward(dLogits);

        double[][] dQ2 = new double[2][2];
        for (int i = 0; i < n; i++) {
            double scale = dBeta[i] / KsqAnchors.frobNormSq(i);
            Mat2.addScaledInPlace(dQ2, KsqAnchors.matrix(i), scale);
        }

        double[][] qT = Mat2.transpose(cachedQ);
        double[][] dQfromLeft = Mat2.mul(dQ2, qT);
        double[][] dQfromRight = Mat2.mul(qT, dQ2);
        double[][] dQ = new double[2][2];
        Mat2.addScaledInPlace(dQ, dQfromLeft, 1.0);
        Mat2.addScaledInPlace(dQ, dQfromRight, 1.0);

        double[] dAlpha = new double[n];
        for (int i = 0; i < n; i++) {
            dAlpha[i] = Mat2.frobInner(dQ, KsqAnchors.matrix(i));
        }

        double[] dSumLogits = tanhBackward(cachedAlpha, dAlpha);

        for (int t = 0; t < cachedTokens.length; t++) {
            embedding.accumulateGradient(cachedTokens[t], dSumLogits);
        }
    }

    /**
     * Subalgebra-specialization regularizer for signed α:
     * λ · Σ_v Σ_{i≠j} α_v[i]² α_v[j]²  =  λ · Σ_v [ (Σ_i α_v[i]²)² − Σ_i α_v[i]⁴ ].
     *
     * <p>Always ≥ 0 (every term is a product of squares). Reaches zero exactly
     * when at most one α_v[i] is non-zero — i.e., clean single-anchor
     * concentration, regardless of sign. Replaces the softmax-era penalty
     * Σ_{i≠j} α_v[i] α_v[j], which under signed α would have a NEGATIVE
     * minimum at α = (1, −1, 0, 0) and so reward mixed-sign two-anchor
     * configurations — the opposite of the specialization mandate.
     */
    public double regularizerLoss(double lambda) {
        if (lambda == 0.0) return 0.0;
        int n = KsqAnchors.N;
        double r = 0.0;
        for (int v = 0; v < embedding.vocab(); v++) {
            double[] alpha = tanh(embedding.lookup(v));
            double sumSq = 0.0;
            double sumQuad = 0.0;
            for (int i = 0; i < n; i++) {
                double a2 = alpha[i] * alpha[i];
                sumSq += a2;
                sumQuad += a2 * a2;
            }
            r += sumSq * sumSq - sumQuad;
        }
        return lambda * r;
    }

    /**
     * Accumulates the regularizer gradient into the embedding table's gradient
     * slots. ∂r_v/∂α_v[k] = 4 · α_v[k] · Σ_{j≠k} α_v[j]² (derived from
     * r_v = (Σ α²)² − Σ α⁴). Then tanh backward maps dα → dLogits via
     * the factor (1 − α_v[k]²).
     */
    public void regularizerBackward(double lambda) {
        if (lambda == 0.0) return;
        int n = KsqAnchors.N;
        for (int v = 0; v < embedding.vocab(); v++) {
            double[] alpha = tanh(embedding.lookup(v));
            double sumSq = 0.0;
            for (int i = 0; i < n; i++) sumSq += alpha[i] * alpha[i];
            double[] dAlpha = new double[n];
            for (int k = 0; k < n; k++) {
                double sumOtherSq = sumSq - alpha[k] * alpha[k];
                dAlpha[k] = 4.0 * lambda * alpha[k] * sumOtherSq;
            }
            embedding.accumulateGradient(v, tanhBackward(alpha, dAlpha));
        }
    }

    /**
     * Cross-vocab contrastive regularizer:
     * ν · Σ_{v1 ≠ v2} ⟨α(v1), α(v2)⟩²  where  α(v) = tanh(E[v]).
     *
     * <p>Always ≥ 0. Zero exactly when every pair of token rows is orthogonal
     * in α-space — i.e., distinct tokens commit to distinct anchors. The
     * per-vocab specialization regularizer {@link #regularizerLoss(double)} is
     * myopic and has no signal to push <i>different</i> tokens toward
     * <i>different</i> anchors; this term supplies exactly that signal.
     * Names the same mandate the K_∞=K_∞ / K_0=K_0 same-anchor collapse
     * failures of iters 3 and 4 named.
     */
    public double crossVocabRegularizerLoss(double nu) {
        if (nu == 0.0) return 0.0;
        int V = embedding.vocab();
        int n = KsqAnchors.N;
        double[][] alphas = new double[V][];
        for (int v = 0; v < V; v++) alphas[v] = tanh(embedding.lookup(v));
        double r = 0.0;
        for (int v1 = 0; v1 < V; v1++) {
            for (int v2 = 0; v2 < V; v2++) {
                if (v1 == v2) continue;
                double c = 0.0;
                for (int i = 0; i < n; i++) c += alphas[v1][i] * alphas[v2][i];
                r += c * c;
            }
        }
        return nu * r;
    }

    /**
     * Accumulates the cross-vocab regularizer gradient into the embedding
     * table. ∂R/∂α_a[k] = 4ν · Σ_{b ≠ a} c(a, b) · α_b[k] (derived from the
     * symmetric ordered-pair sum and chain rule on the squared inner
     * product). Then tanh backward via the factor (1 − α_a[k]²).
     */
    public void crossVocabRegularizerBackward(double nu) {
        if (nu == 0.0) return;
        int V = embedding.vocab();
        int n = KsqAnchors.N;
        double[][] alphas = new double[V][];
        for (int v = 0; v < V; v++) alphas[v] = tanh(embedding.lookup(v));

        double[][] c = new double[V][V];
        for (int v1 = 0; v1 < V; v1++) {
            for (int v2 = 0; v2 < V; v2++) {
                if (v1 == v2) continue;
                double s = 0.0;
                for (int i = 0; i < n; i++) s += alphas[v1][i] * alphas[v2][i];
                c[v1][v2] = s;
            }
        }

        for (int va = 0; va < V; va++) {
            double[] dAlpha = new double[n];
            for (int vb = 0; vb < V; vb++) {
                if (va == vb) continue;
                double scale = 4.0 * nu * c[va][vb];
                for (int k = 0; k < n; k++) {
                    dAlpha[k] += scale * alphas[vb][k];
                }
            }
            embedding.accumulateGradient(va, tanhBackward(alphas[va], dAlpha));
        }
    }

    public void step(double lr) {
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

    private static double[] softmaxBackward(double[] alpha, double[] dAlpha) {
        int n = alpha.length;
        double dot = 0.0;
        for (int i = 0; i < n; i++) dot += dAlpha[i] * alpha[i];
        double[] dLogits = new double[n];
        for (int j = 0; j < n; j++) {
            dLogits[j] = alpha[j] * (dAlpha[j] - dot);
        }
        return dLogits;
    }

    private static double[] tanh(double[] logits) {
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) out[i] = Math.tanh(logits[i]);
        return out;
    }

    /**
     * Threshold used by demos to distinguish "K_0 cleanly dominates and
     * other anchors are inactive" (truly scalar specialization, algebra
     * collapsed to R) from "K_0 happens to be slightly largest but other
     * anchors are also strong" (saturated everywhere, algebra still
     * non-trivially used).
     */
    public static final double SCALAR_OTHER_THRESHOLD = 0.5;

    private static double[] tanhBackward(double[] alpha, double[] dAlpha) {
        double[] dLogits = new double[alpha.length];
        for (int i = 0; i < alpha.length; i++) {
            dLogits[i] = dAlpha[i] * (1.0 - alpha[i] * alpha[i]);
        }
        return dLogits;
    }

    private static double[][] anchorCombination(double[] alpha) {
        double[][] q = new double[2][2];
        for (int i = 0; i < KsqAnchors.N; i++) {
            Mat2.addScaledInPlace(q, KsqAnchors.matrix(i), alpha[i]);
        }
        return q;
    }

    private static double[] anchorProjection(double[][] s) {
        double[] beta = new double[KsqAnchors.N];
        for (int i = 0; i < KsqAnchors.N; i++) {
            beta[i] = Mat2.frobInner(s, KsqAnchors.matrix(i)) / KsqAnchors.frobNormSq(i);
        }
        return beta;
    }
}
