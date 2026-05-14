package sibarum.strnn.ksqp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * KSQP — Keyed Split-Quaternion with polynomial lift, conjugate sandwich,
 * and discrete-degree control by null-cone events.
 *
 * <p>Per-token forward pipeline:
 *
 * <pre>
 *   v ← token index
 *   k_v ∈ ℝⁿ        : fixed random per-token input (frozen at construction)
 *   sq_v ∈ ℝ⁴       : split-quaternion parameters (gradient-trained)
 *   p_v ∈ ℤ⁺        : polynomial degree (event-driven, discrete)
 *
 *   m  = PolyLift_p_v(k_v)              monomial vector, length C(n+p-1, p)
 *   x  = P_{p_v} · m                    learned projection down to ℝ⁴
 *   y  = sq_v · x · sq_v̄                conjugate sandwich, ℝ⁴ → ℝ⁴
 *
 *   y_seq = y_0 · y_1 · … · y_{T-1}     split-quat product across tokens
 *   logits = W · y_seq + b              linear head, W ∈ ℝ^{outDim × 4}
 * </pre>
 *
 * <p>Aggregation choice: <b>left-to-right split-quat product</b> across
 * the per-token outputs. Sum-pool puts the four XOR inputs at
 * {2y_0, y_0+y_1, y_0+y_1, 2y_1} — three collinear points, un-separable
 * by any linear head. Concat ([y_t, y_t']) makes the head's logit
 * additively linear in the two token features, which cannot represent
 * XOR (one can show the required inequalities directly contradict each
 * other). Split-quat product introduces a non-commutative bilinear
 * cross-token interaction inside the algebra: the four XOR sequences
 * map to four distinct points in ℝ⁴, and the head can separate them.
 * Generalizes to any seqLen ≥ 1 without resizing the head.
 *
 * <p>Null-cone event detection: per the plan, "no hysteresis at this
 * stage, just choose a reasonable threshold for both sides of the null
 * cone." Per-token: track sign of N(sq_v) across steps. A sign flip is
 * a crossing event:
 * <ul>
 *   <li>N: + → -  ⇒ <b>increment</b> p_v</li>
 *   <li>N: - → +  ⇒ <b>decrement</b> p_v</li>
 * </ul>
 * Mapping is TBD per the plan; the new architecture restarts the A/B
 * fresh because the prior measurements were on a model missing the
 * vector×SQ pathway entirely. p ∈ [P_MIN, P_MAX] (clamped). On event:
 * sq_v is restored to its initial value.
 */
public final class KsqpModel {

    public static final int SQ_DIM = SplitQuat.DIM;

    public static final int P_INIT = 1;
    public static final int P_MIN = 1;
    public static final int P_MAX = 4;

    public static final double DEFAULT_SQ_INIT_A0 = 0.5;
    public static final double DEFAULT_SQ_INIT_NOISE = 0.1;
    public static final double DEFAULT_K_INIT_RANGE = 0.5;

    private final int vocab;
    private final int outDim;
    private final int seqLen;
    private final int n;

    private final double[][] k;           // [vocab][n]   — FROZEN
    private final double[][] sq;          // [vocab][4]
    private final double[][] sqInit;      // [vocab][4]   — snapshot for re-init
    private final double[][] sqGrad;      // [vocab][4]
    private final int[] p;                // [vocab]
    private final int[] prevSign;         // [vocab]

    private final int[][][] monomials;            // [d - P_MIN][monomial_row][n]
    private final double[][][] projection;        // [d - P_MIN][monomial_row][4]
    private final double[][][] projectionGrad;    // [d - P_MIN][monomial_row][4]

    private final double[][] headW;       // [outDim][4]
    private final double[] headB;         // [outDim]
    private final double[][] headWGrad;
    private final double[] headBGrad;

    // Forward-pass cache.
    private int[] cachedTokens;
    private double[][] cachedLifted;     // [pos][monomial_row]
    private double[][] cachedProjected;  // [pos][4]
    private double[][] cachedY;          // [pos][4]   per-token sandwich outputs
    private double[][] cachedPartial;    // [pos][4]   running split-quat product:
                                         //            cachedPartial[t] = y_0 · y_1 · … · y_t
    private double[] cachedYSeq;         // alias for cachedPartial[seqLen-1]
    private double[] cachedLogits;

    public KsqpModel(int vocab, int outDim, int seqLen, int n, long seed) {
        this(vocab, outDim, seqLen, n, seed,
                DEFAULT_SQ_INIT_A0, DEFAULT_SQ_INIT_NOISE, DEFAULT_K_INIT_RANGE);
    }

    public KsqpModel(int vocab, int outDim, int seqLen, int n, long seed,
                     double sqInitA0, double sqInitNoise, double kInitRange) {
        this.vocab = vocab;
        this.outDim = outDim;
        this.seqLen = seqLen;
        this.n = n;

        Random rng = new Random(seed);

        this.k = new double[vocab][n];
        for (int v = 0; v < vocab; v++) {
            for (int j = 0; j < n; j++) {
                k[v][j] = (rng.nextDouble() * 2 - 1) * kInitRange;
            }
        }

        this.sq = new double[vocab][SQ_DIM];
        this.sqInit = new double[vocab][SQ_DIM];
        this.sqGrad = new double[vocab][SQ_DIM];
        this.p = new int[vocab];
        this.prevSign = new int[vocab];
        for (int v = 0; v < vocab; v++) {
            sq[v][0] = sqInitA0 + (rng.nextDouble() * 2 - 1) * sqInitNoise;
            for (int a = 1; a < SQ_DIM; a++) {
                sq[v][a] = (rng.nextDouble() * 2 - 1) * sqInitNoise;
            }
            System.arraycopy(sq[v], 0, sqInit[v], 0, SQ_DIM);
            p[v] = P_INIT;
            prevSign[v] = (int) Math.signum(SplitQuat.norm(sq[v]));
        }

        int degreeRange = P_MAX - P_MIN + 1;
        this.monomials = new int[degreeRange][][];
        this.projection = new double[degreeRange][][];
        this.projectionGrad = new double[degreeRange][][];
        for (int dIdx = 0; dIdx < degreeRange; dIdx++) {
            int d = P_MIN + dIdx;
            monomials[dIdx] = PolyLift.enumerateMonomials(n, d);
            int M = monomials[dIdx].length;
            projection[dIdx] = new double[M][SQ_DIM];
            projectionGrad[dIdx] = new double[M][SQ_DIM];
            double bound = 1.0 / Math.sqrt(M);
            for (int r = 0; r < M; r++) {
                for (int c = 0; c < SQ_DIM; c++) {
                    projection[dIdx][r][c] = (rng.nextDouble() * 2 - 1) * bound;
                }
            }
        }

        this.headW = new double[outDim][SQ_DIM];
        this.headB = new double[outDim];
        this.headWGrad = new double[outDim][SQ_DIM];
        this.headBGrad = new double[outDim];
        double headBound = 1.0 / Math.sqrt(SQ_DIM);
        for (int o = 0; o < outDim; o++) {
            for (int i = 0; i < SQ_DIM; i++) {
                headW[o][i] = (rng.nextDouble() * 2 - 1) * headBound;
            }
        }
    }

    public int vocab() { return vocab; }
    public int outDim() { return outDim; }
    public int seqLen() { return seqLen; }
    public int n() { return n; }
    public double[] k(int v) { return k[v]; }
    public double[] sq(int v) { return sq[v]; }
    public int p(int v) { return p[v]; }
    public double[][] headWeights() { return headW; }
    public double[] headBiases() { return headB; }
    public double[][] sqGradients() { return sqGrad; }
    public double[][] headWGradients() { return headWGrad; }
    public double[] headBGradients() { return headBGrad; }
    public double[][] projectionFor(int d) { return projection[d - P_MIN]; }
    public double[][] projectionGradFor(int d) { return projectionGrad[d - P_MIN]; }
    public int[][] monomialsFor(int d) { return monomials[d - P_MIN]; }

    public void setP(int v, int newP) {
        if (newP < P_MIN || newP > P_MAX) {
            throw new IllegalArgumentException("p out of bounds: " + newP);
        }
        p[v] = newP;
    }

    public void zeroGrad() {
        for (int v = 0; v < vocab; v++) {
            for (int a = 0; a < SQ_DIM; a++) sqGrad[v][a] = 0.0;
        }
        for (int dIdx = 0; dIdx < projection.length; dIdx++) {
            for (int r = 0; r < projection[dIdx].length; r++) {
                for (int c = 0; c < SQ_DIM; c++) projectionGrad[dIdx][r][c] = 0.0;
            }
        }
        for (int o = 0; o < outDim; o++) {
            for (int i = 0; i < headW[o].length; i++) headWGrad[o][i] = 0.0;
            headBGrad[o] = 0.0;
        }
    }

    public double[] forward(int[] tokens) {
        if (tokens == null || tokens.length != seqLen) {
            throw new IllegalArgumentException("tokens length must equal seqLen " + seqLen);
        }
        cachedTokens = tokens.clone();
        cachedLifted = new double[seqLen][];
        cachedProjected = new double[seqLen][];
        cachedY = new double[seqLen][];
        cachedPartial = new double[seqLen][];

        for (int pos = 0; pos < seqLen; pos++) {
            int v = tokens[pos];
            int d = p[v];
            int dIdx = d - P_MIN;
            cachedLifted[pos] = PolyLift.lift(k[v], monomials[dIdx]);
            cachedProjected[pos] = new double[SQ_DIM];
            int M = cachedLifted[pos].length;
            for (int r = 0; r < M; r++) {
                double mr = cachedLifted[pos][r];
                for (int i = 0; i < SQ_DIM; i++) {
                    cachedProjected[pos][i] += projection[dIdx][r][i] * mr;
                }
            }
            cachedY[pos] = SplitQuat.sandwich(sq[v], cachedProjected[pos]);
            cachedPartial[pos] = (pos == 0)
                    ? cachedY[pos].clone()
                    : SplitQuat.mul(cachedPartial[pos - 1], cachedY[pos]);
        }
        cachedYSeq = cachedPartial[seqLen - 1];

        cachedLogits = new double[outDim];
        for (int o = 0; o < outDim; o++) {
            double acc = headB[o];
            for (int i = 0; i < SQ_DIM; i++) acc += headW[o][i] * cachedYSeq[i];
            cachedLogits[o] = acc;
        }
        return cachedLogits.clone();
    }

    public double crossEntropyLoss(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");
        double m = Double.NEGATIVE_INFINITY;
        for (double val : cachedLogits) if (val > m) m = val;
        double sumExp = 0.0;
        for (double val : cachedLogits) sumExp += Math.exp(val - m);
        return Math.log(sumExp) + m - cachedLogits[target];
    }

    public void backward(int target) {
        if (cachedLogits == null) throw new IllegalStateException("forward must be called first");

        double[] dLogits = softmax(cachedLogits);
        dLogits[target] -= 1.0;

        double[] dYSeq = new double[SQ_DIM];
        for (int o = 0; o < outDim; o++) {
            headBGrad[o] += dLogits[o];
            for (int i = 0; i < SQ_DIM; i++) {
                headWGrad[o][i] += dLogits[o] * cachedYSeq[i];
                dYSeq[i] += dLogits[o] * headW[o][i];
            }
        }

        // Backward through the split-quat product chain
        //   partial_0 = y_0
        //   partial_t = partial_{t-1} · y_t           (t ≥ 1)
        //   y_seq     = partial_{seqLen-1}
        // Walk the chain right-to-left: at step t we hold dL/dpartial_t and
        // split it into dL/dpartial_{t-1} (via mulBackwardA on right operand y_t)
        // and dL/dy_t (via mulBackwardB on left operand partial_{t-1}).
        double[] dPartial = dYSeq;
        double[][] dY = new double[seqLen][];
        for (int t = seqLen - 1; t >= 1; t--) {
            dY[t] = SplitQuat.mulBackwardB(cachedPartial[t - 1], dPartial);
            dPartial = SplitQuat.mulBackwardA(cachedY[t], dPartial);
        }
        dY[0] = dPartial; // remaining gradient flows directly into y_0 (= partial_0)

        for (int pos = 0; pos < seqLen; pos++) {
            int v = cachedTokens[pos];
            int d = p[v];
            int dIdx = d - P_MIN;

            double[][] dSqdV = SplitQuat.sandwichBackward(sq[v], cachedProjected[pos], dY[pos]);
            double[] dSq = dSqdV[0];
            double[] dProjected = dSqdV[1];
            for (int a = 0; a < SQ_DIM; a++) sqGrad[v][a] += dSq[a];

            int M = cachedLifted[pos].length;
            for (int r = 0; r < M; r++) {
                double mr = cachedLifted[pos][r];
                for (int i = 0; i < SQ_DIM; i++) {
                    projectionGrad[dIdx][r][i] += dProjected[i] * mr;
                }
            }
            // dM (and thus dK) intentionally not propagated — k is frozen.
        }
    }

    public void step(double lr) {
        for (int v = 0; v < vocab; v++) {
            for (int a = 0; a < SQ_DIM; a++) sq[v][a] -= lr * sqGrad[v][a];
        }
        for (int dIdx = 0; dIdx < projection.length; dIdx++) {
            for (int r = 0; r < projection[dIdx].length; r++) {
                for (int c = 0; c < SQ_DIM; c++) {
                    projection[dIdx][r][c] -= lr * projectionGrad[dIdx][r][c];
                }
            }
        }
        for (int o = 0; o < outDim; o++) {
            for (int i = 0; i < headW[o].length; i++) headW[o][i] -= lr * headWGrad[o][i];
            headB[o] -= lr * headBGrad[o];
        }
    }

    public record EventRecord(int epoch, int tokenId, int prevP, int newP,
                              double normBefore, double normAfter,
                              double[] sqAtEvent) {}

    public List<EventRecord> detectEvents(int epoch) {
        List<EventRecord> events = new ArrayList<>();
        for (int v = 0; v < vocab; v++) {
            double normNow = SplitQuat.norm(sq[v]);
            int signNow = (int) Math.signum(normNow);
            if (signNow == 0) signNow = prevSign[v];
            int signPrev = prevSign[v];

            if (signPrev != 0 && signNow != 0 && signPrev != signNow) {
                int prevP = p[v];
                int newP;
                if (signPrev > 0 && signNow < 0) {
                    newP = Math.min(P_MAX, prevP + 1);
                } else {
                    newP = Math.max(P_MIN, prevP - 1);
                }
                double[] sqAtEvent = sq[v].clone();
                p[v] = newP;
                System.arraycopy(sqInit[v], 0, sq[v], 0, SQ_DIM);
                double normAfter = SplitQuat.norm(sq[v]);
                prevSign[v] = (int) Math.signum(normAfter);
                events.add(new EventRecord(epoch, v, prevP, newP, normNow, normAfter, sqAtEvent));
            } else {
                prevSign[v] = signNow;
            }
        }
        return events;
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
}
