package sibarum.strnn.ksqp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * KSQP with a content-addressable key-value cache: M stored entries are
 * retrieved by similarity to a continuous query vector, not by index.
 * Mirrors {@link KsqpModel}'s per-entry mechanics (lift → project →
 * sandwich, plus the discrete-degree null-cone event machinery) but
 * the aggregation is a soft-attention pool over the M entries rather
 * than a token-product chain.
 *
 * <p>Forward pass for a single query q ∈ ℝ^queryDim:
 *
 * <pre>
 *   sim_v   = − ‖q − storedKeys_v‖² / τ            (negative squared distance)
 *   w       = softmax(sim)                          (length M)
 *
 *   per entry v ∈ 0..M-1:
 *     m_v   = lift(q, monomials_{p_v})              all degree-p_v monomials of q
 *     x_v   = P_{p_v} · m_v                         learned projection to ℝ⁴
 *     y_v   = sq_v · x_v · sq̄_v                    conjugate sandwich
 *
 *   y_agg   = Σ_v w_v · y_v                         attention pool over the KV
 *   logits  = W · y_agg + b                         linear head
 * </pre>
 *
 * <p>Stored keys can be either frozen (passed in at construction, no
 * gradient) or trainable (initialized randomly at construction, learned
 * by gradient descent through the softmax-attention pathway). Toggle
 * via the {@code trainStoredKeys} constructor argument. sq_v and the
 * per-degree projection matrices are always gradient-trained; p_v is
 * moved by the null-cone event mechanism.
 *
 * <p>Query is treated as input data and never gradient-trained.
 */
public final class KsqpKvModel {

    public static final int SQ_DIM = SplitQuat.DIM;

    public static final int P_INIT = 1;
    public static final int P_MIN = 1;
    public static final int P_MAX = 4;

    public static final double DEFAULT_SQ_INIT_A0 = 0.5;
    public static final double DEFAULT_SQ_INIT_NOISE = 0.1;
    public static final double DEFAULT_TEMPERATURE = 1.0;
    public static final double DEFAULT_STORED_KEY_INIT_RANGE = 0.5;

    private final int M;
    private final int outDim;
    private final int queryDim;
    private final double temperature;
    private final boolean trainStoredKeys;

    private final double[][] storedKeys;       // [M][queryDim]
    private final double[][] storedKeysGrad;   // [M][queryDim] — null when not trainable
    private final double[][] sq;             // [M][4]
    private final double[][] sqInit;         // [M][4]
    private final double[][] sqGrad;         // [M][4]
    private final int[] p;                   // [M]
    private final int[] prevSign;            // [M]

    private final int[][][] monomials;            // [d-P_MIN][monomial_row][queryDim]
    private final double[][][] projection;        // [d-P_MIN][monomial_row][4]
    private final double[][][] projectionGrad;

    private final double[][] headW;          // [outDim][4]
    private final double[] headB;            // [outDim]
    private final double[][] headWGrad;
    private final double[] headBGrad;

    // Forward-pass cache.
    private double[] cachedQuery;
    private double[] cachedSims;
    private double[] cachedWeights;          // softmax over sims, length M
    private double[][] cachedLifted;         // [M][monomial_count]
    private double[][] cachedProjected;      // [M][4]
    private double[][] cachedY;              // [M][4]
    private double[] cachedYAgg;             // length 4
    private double[] cachedLogits;

    /** Frozen stored keys — keys are provided and held constant. */
    public KsqpKvModel(int outDim, int queryDim, double[][] storedKeys, long seed) {
        this(outDim, queryDim, storedKeys, seed,
                DEFAULT_SQ_INIT_A0, DEFAULT_SQ_INIT_NOISE, DEFAULT_TEMPERATURE);
    }

    /** Frozen stored keys — keys are provided and held constant. */
    public KsqpKvModel(int outDim, int queryDim, double[][] storedKeys, long seed,
                       double sqInitA0, double sqInitNoise, double temperature) {
        this(outDim, queryDim, storedKeys, /*trainStoredKeys=*/false, seed,
                sqInitA0, sqInitNoise, temperature);
    }

    /**
     * Trainable stored keys — initialized random uniform in
     * [−storedKeyInitRange, +storedKeyInitRange] and gradient-trained.
     */
    public KsqpKvModel(int outDim, int queryDim, int M, long seed,
                       double sqInitA0, double sqInitNoise, double temperature,
                       double storedKeyInitRange) {
        this(outDim, queryDim, randomKeys(M, queryDim, seed, storedKeyInitRange),
                /*trainStoredKeys=*/true, seed, sqInitA0, sqInitNoise, temperature);
    }

    private KsqpKvModel(int outDim, int queryDim, double[][] storedKeys,
                        boolean trainStoredKeys, long seed,
                        double sqInitA0, double sqInitNoise, double temperature) {
        if (storedKeys == null || storedKeys.length == 0) {
            throw new IllegalArgumentException("storedKeys must be non-empty");
        }
        for (double[] row : storedKeys) {
            if (row.length != queryDim) {
                throw new IllegalArgumentException("each stored key must have length queryDim");
            }
        }
        this.M = storedKeys.length;
        this.outDim = outDim;
        this.queryDim = queryDim;
        this.temperature = temperature;
        this.trainStoredKeys = trainStoredKeys;

        this.storedKeys = new double[M][queryDim];
        for (int v = 0; v < M; v++) {
            System.arraycopy(storedKeys[v], 0, this.storedKeys[v], 0, queryDim);
        }
        this.storedKeysGrad = trainStoredKeys ? new double[M][queryDim] : null;

        Random rng = new Random(seed);

        this.sq = new double[M][SQ_DIM];
        this.sqInit = new double[M][SQ_DIM];
        this.sqGrad = new double[M][SQ_DIM];
        this.p = new int[M];
        this.prevSign = new int[M];
        for (int v = 0; v < M; v++) {
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
            monomials[dIdx] = PolyLift.enumerateMonomials(queryDim, d);
            int Mmon = monomials[dIdx].length;
            projection[dIdx] = new double[Mmon][SQ_DIM];
            projectionGrad[dIdx] = new double[Mmon][SQ_DIM];
            double bound = 1.0 / Math.sqrt(Mmon);
            for (int r = 0; r < Mmon; r++) {
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

    public int entries() { return M; }
    public int outDim() { return outDim; }
    public int queryDim() { return queryDim; }
    public double temperature() { return temperature; }
    public boolean trainStoredKeys() { return trainStoredKeys; }
    public double[] storedKey(int v) { return storedKeys[v]; }
    public double[] sq(int v) { return sq[v]; }
    public int p(int v) { return p[v]; }
    public double[][] headWeights() { return headW; }
    public double[] headBiases() { return headB; }
    public double[][] sqGradients() { return sqGrad; }
    public double[][] headWGradients() { return headWGrad; }
    public double[] headBGradients() { return headBGrad; }
    public double[][] storedKeysGradients() { return storedKeysGrad; }
    public double[][] projectionFor(int d) { return projection[d - P_MIN]; }
    public double[][] projectionGradFor(int d) { return projectionGrad[d - P_MIN]; }
    public double[] lastWeights() { return cachedWeights; }

    public void setP(int v, int newP) {
        if (newP < P_MIN || newP > P_MAX) {
            throw new IllegalArgumentException("p out of bounds: " + newP);
        }
        p[v] = newP;
    }

    public void zeroGrad() {
        for (int v = 0; v < M; v++) {
            for (int a = 0; a < SQ_DIM; a++) sqGrad[v][a] = 0.0;
        }
        for (int dIdx = 0; dIdx < projection.length; dIdx++) {
            for (int r = 0; r < projection[dIdx].length; r++) {
                for (int c = 0; c < SQ_DIM; c++) projectionGrad[dIdx][r][c] = 0.0;
            }
        }
        for (int o = 0; o < outDim; o++) {
            for (int i = 0; i < SQ_DIM; i++) headWGrad[o][i] = 0.0;
            headBGrad[o] = 0.0;
        }
        if (trainStoredKeys) {
            for (int v = 0; v < M; v++) {
                for (int j = 0; j < queryDim; j++) storedKeysGrad[v][j] = 0.0;
            }
        }
    }

    public double[] forward(double[] query) {
        if (query == null || query.length != queryDim) {
            throw new IllegalArgumentException("query length must equal queryDim " + queryDim);
        }
        cachedQuery = query.clone();

        cachedSims = new double[M];
        for (int v = 0; v < M; v++) {
            double s = 0.0;
            for (int j = 0; j < queryDim; j++) {
                double diff = query[j] - storedKeys[v][j];
                s += diff * diff;
            }
            cachedSims[v] = -s / temperature;
        }
        cachedWeights = softmax(cachedSims);

        cachedLifted = new double[M][];
        cachedProjected = new double[M][SQ_DIM];
        cachedY = new double[M][];
        cachedYAgg = new double[SQ_DIM];
        for (int v = 0; v < M; v++) {
            int d = p[v];
            int dIdx = d - P_MIN;
            cachedLifted[v] = PolyLift.lift(query, monomials[dIdx]);
            int Mmon = cachedLifted[v].length;
            for (int r = 0; r < Mmon; r++) {
                double mr = cachedLifted[v][r];
                for (int i = 0; i < SQ_DIM; i++) {
                    cachedProjected[v][i] += projection[dIdx][r][i] * mr;
                }
            }
            cachedY[v] = SplitQuat.sandwich(sq[v], cachedProjected[v]);
            double w = cachedWeights[v];
            for (int i = 0; i < SQ_DIM; i++) cachedYAgg[i] += w * cachedY[v][i];
        }

        cachedLogits = new double[outDim];
        for (int o = 0; o < outDim; o++) {
            double acc = headB[o];
            for (int i = 0; i < SQ_DIM; i++) acc += headW[o][i] * cachedYAgg[i];
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

        // Head: logits = W · yAgg + b.
        double[] dYAgg = new double[SQ_DIM];
        for (int o = 0; o < outDim; o++) {
            headBGrad[o] += dLogits[o];
            for (int i = 0; i < SQ_DIM; i++) {
                headWGrad[o][i] += dLogits[o] * cachedYAgg[i];
                dYAgg[i] += dLogits[o] * headW[o][i];
            }
        }

        // yAgg = Σ_v w_v · y_v   ⇒   dL/dw_v = ⟨dYAgg, y_v⟩,  dL/dy_v = w_v · dYAgg.
        double[] dWeights = new double[M];
        double[][] dYper = new double[M][SQ_DIM];
        for (int v = 0; v < M; v++) {
            double dw = 0.0;
            for (int i = 0; i < SQ_DIM; i++) {
                dw += dYAgg[i] * cachedY[v][i];
                dYper[v][i] = cachedWeights[v] * dYAgg[i];
            }
            dWeights[v] = dw;
        }

        // Softmax backward: dSim = (J_softmax)^T · dWeights, where
        //   ∂w_a/∂sim_b = w_a (δ_ab − w_b).
        // Closed form: dSim_b = w_b · (dWeights_b − ⟨w, dWeights⟩).
        if (trainStoredKeys) {
            double dotWDw = 0.0;
            for (int v = 0; v < M; v++) dotWDw += cachedWeights[v] * dWeights[v];
            // sim_v = − ‖q − k_v‖² / τ   ⇒   ∂sim_v/∂k_v[j] = +2 (q_j − k_v[j]) / τ.
            for (int v = 0; v < M; v++) {
                double dSimV = cachedWeights[v] * (dWeights[v] - dotWDw);
                double scale = 2.0 * dSimV / temperature;
                for (int j = 0; j < queryDim; j++) {
                    storedKeysGrad[v][j] += scale * (cachedQuery[j] - storedKeys[v][j]);
                }
            }
        }
        // (When stored keys are frozen, dSim isn't propagated anywhere — query is input data.)

        // Per-entry: sandwich and projection backward, scaled by w_v through dYper.
        for (int v = 0; v < M; v++) {
            int d = p[v];
            int dIdx = d - P_MIN;

            double[][] dSqdV = SplitQuat.sandwichBackward(sq[v], cachedProjected[v], dYper[v]);
            double[] dSq = dSqdV[0];
            double[] dProjected = dSqdV[1];
            for (int a = 0; a < SQ_DIM; a++) sqGrad[v][a] += dSq[a];

            int Mmon = cachedLifted[v].length;
            for (int r = 0; r < Mmon; r++) {
                double mr = cachedLifted[v][r];
                for (int i = 0; i < SQ_DIM; i++) {
                    projectionGrad[dIdx][r][i] += dProjected[i] * mr;
                }
            }
            // dLifted / dQuery not propagated — query is input data.
        }
    }

    public void step(double lr) {
        for (int v = 0; v < M; v++) {
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
            for (int i = 0; i < SQ_DIM; i++) headW[o][i] -= lr * headWGrad[o][i];
            headB[o] -= lr * headBGrad[o];
        }
        if (trainStoredKeys) {
            for (int v = 0; v < M; v++) {
                for (int j = 0; j < queryDim; j++) {
                    storedKeys[v][j] -= lr * storedKeysGrad[v][j];
                }
            }
        }
    }

    public record EventRecord(int epoch, int entryId, int prevP, int newP,
                              double normBefore, double normAfter,
                              double[] sqAtEvent) {}

    public List<EventRecord> detectEvents(int epoch) {
        List<EventRecord> events = new ArrayList<>();
        for (int v = 0; v < M; v++) {
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

    private static double[][] randomKeys(int M, int queryDim, long seed, double range) {
        Random rng = new Random(seed ^ 0xA5A5_5A5A_C3C3_3C3CL);
        double[][] keys = new double[M][queryDim];
        for (int v = 0; v < M; v++) {
            for (int j = 0; j < queryDim; j++) {
                keys[v][j] = (rng.nextDouble() * 2 - 1) * range;
            }
        }
        return keys;
    }

    private static double[] softmax(double[] xs) {
        int n = xs.length;
        double m = Double.NEGATIVE_INFINITY;
        for (double v : xs) if (v > m) m = v;
        double[] out = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            out[i] = Math.exp(xs[i] - m);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
        return out;
    }
}
