package sibarum.strnn.ksq;

/**
 * The four Möbius cardinals in the split-quaternion algebra H_s, represented
 * via the isomorphism H_s ≅ M_2(R) as 2x2 real matrices.
 *
 * <ul>
 *   <li>K_0 = 1 (scalar identity, R)</li>
 *   <li>K_i = i (elliptic / rotation generator, C)</li>
 *   <li>K_j = j (hyperbolic / boost generator, R[j])</li>
 *   <li>K_inf = e_+ = (1+j)/2 (idempotent / parabolic, in R[j])</li>
 * </ul>
 *
 * <p>K_0, K_i, K_j and K_inf are LINEARLY DEPENDENT in M_2(R):
 * K_0 + K_j = 2·K_inf (since e_+ + e_- = 1 and e_+ - e_- = j gives
 * 2 e_+ = 1 + j). This dependence is intentional: K_j and K_inf are two
 * bases of the same subalgebra R[j] — generator basis (j = e_+ - e_-)
 * vs. idempotent basis (e_+, e_-). The optimizer's choice between
 * K_j-specialization and K_inf-specialization corresponds to "express
 * this token in the generator basis vs. the idempotent basis of the
 * same subalgebra," which is the cleanest algebraic story for the
 * inference-specialization collapse (componentwise multiply in the
 * idempotent basis).
 *
 * <p>The four anchors span 3 dimensions in M_2(R), not 4. The bilinear
 * step Q² still escapes this 3-dim span via cross products like
 * K_i · K_j = ±k (which is NOT in span{K_0, K_i, K_j}), so the readout
 * onto {K_0, K_i, K_j, K_inf} still produces non-trivial features for
 * the head to discriminate.
 *
 * <p>Anchors are fixed (NOT trainable) — they carry the algebra's
 * structure into the model; only the embedding table indexing into them
 * is learned.
 */
public final class KsqAnchors {

    public static final int N = 4;

    public static final int K0 = 0;
    public static final int KI = 1;
    public static final int KJ = 2;
    public static final int KINF = 3;

    private static final double[][][] ANCHORS = new double[][][] {
            { { 1.0, 0.0 }, { 0.0, 1.0 } },
            { { 0.0, 1.0 }, { -1.0, 0.0 } },
            { { 1.0, 0.0 }, { 0.0, -1.0 } },
            { { 1.0, 0.0 }, { 0.0, 0.0 } }
    };

    private static final String[] NAMES = { "K_0", "K_i", "K_j", "K_inf" };

    private static final double[] FROB_NORM_SQ = computeFrobNormSq();

    private KsqAnchors() {}

    public static double[][] matrix(int index) {
        return ANCHORS[index];
    }

    public static String name(int index) {
        return NAMES[index];
    }

    /** Frobenius norm squared of anchor {@code index}: sum of squared entries. */
    public static double frobNormSq(int index) {
        return FROB_NORM_SQ[index];
    }

    private static double[] computeFrobNormSq() {
        double[] out = new double[N];
        for (int n = 0; n < N; n++) {
            double s = 0.0;
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    s += ANCHORS[n][a][b] * ANCHORS[n][a][b];
                }
            }
            out[n] = s;
        }
        return out;
    }
}
