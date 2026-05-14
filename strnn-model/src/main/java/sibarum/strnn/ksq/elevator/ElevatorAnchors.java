package sibarum.strnn.ksq.elevator;

/**
 * Elevator-architecture anchors. Same Möbius cardinals as iter-5 KSQ in
 * {@link sibarum.strnn.ksq.KsqAnchors}, separated into its own package so
 * Phase 3 can add the projective anchor K_eMinus without modifying the
 * iter-5 KSQ. For Phases 1 and 2, this anchor set is identical to iter-5.
 *
 * <ul>
 *   <li>K_0 = 1 (scalar identity, R)</li>
 *   <li>K_i = i (elliptic / rotation generator, C)</li>
 *   <li>K_j = j (hyperbolic / boost generator, R[j])</li>
 *   <li>K_inf = e_+ = (1+j)/2 (idempotent / parabolic, in R[j])</li>
 *   <li>K_eMinus (Phase 3): e_- = (1-j)/2 (idempotent / projective-placeholder, in R[j])</li>
 * </ul>
 */
public final class ElevatorAnchors {

    public static final int N = 5;

    public static final int K0 = 0;
    public static final int KI = 1;
    public static final int KJ = 2;
    public static final int KINF = 3;
    public static final int K_E_MINUS = 4;

    private static final double[][][] ANCHORS = new double[][][] {
            { { 1.0, 0.0 }, { 0.0, 1.0 } },
            { { 0.0, 1.0 }, { -1.0, 0.0 } },
            { { 1.0, 0.0 }, { 0.0, -1.0 } },
            { { 1.0, 0.0 }, { 0.0, 0.0 } },
            { { 0.0, 0.0 }, { 0.0, 1.0 } }
    };

    private static final String[] NAMES = { "K_0", "K_i", "K_j", "K_inf", "K_eMinus" };

    private static final double[] FROB_NORM_SQ = computeFrobNormSq();

    private ElevatorAnchors() {}

    public static double[][] matrix(int index) {
        return ANCHORS[index];
    }

    public static String name(int index) {
        return NAMES[index];
    }

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
