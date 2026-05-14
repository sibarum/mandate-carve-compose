package sibarum.mcc.op.advanced;

import sibarum.mcc.value.QuaternionValue;

/**
 * Split-quaternion algebra on 4-vectors in the basis {@code {1, i, j, k}}
 * with relations:
 *
 * <pre>
 *   i² = −1
 *   j² = +1
 *   k² = +1
 *   ij = k, ji = −k
 *   jk = −i, kj = i
 *   ki = j, ik = −j
 * </pre>
 *
 * <p>Norm form: {@code N(a) = a₀² + a₁² − a₂² − a₃²} (signature
 * {@code (++--)}). The canonical idempotents are
 * {@code e_+ = (1/2, 0, 1/2, 0)} and {@code e_- = (1/2, 0, −1/2, 0)};
 * both lie on the null cone {@code {N = 0}}.
 *
 * <p>Fast inner-loop API uses bare {@code double[4]} for performance.
 * The {@link #fromValue} and {@link #toValue} converters bridge to
 * {@link QuaternionValue} when a primitive needs to surface results
 * to a graph.
 */
public final class SplitQuat {

    public static final int DIM = 4;

    private SplitQuat() {}

    /** Convert a {@link QuaternionValue} to the bare {@code double[4]} form. */
    public static double[] fromValue(QuaternionValue q) {
        return q.toArray();
    }

    /** Wrap a {@code double[4]} as a {@link QuaternionValue}. */
    public static QuaternionValue toValue(double[] a) {
        return new QuaternionValue(a);
    }

    /**
     * Full split-quaternion product {@code (a · b)} in 4-vector form.
     * See the multiplication table at the top of the file.
     */
    public static double[] mul(double[] a, double[] b) {
        double[] c = new double[DIM];
        c[0] = a[0]*b[0] - a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
        c[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[3] + a[3]*b[2];
        c[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[3] + a[3]*b[1];
        c[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1];
        return c;
    }

    /**
     * Split-quaternion conjugate: scalar part unchanged, i/j/k parts
     * negated.
     */
    public static double[] conj(double[] a) {
        return new double[] { a[0], -a[1], -a[2], -a[3] };
    }

    /**
     * Backward of {@code c = a · b} w.r.t. the left operand. Given
     * {@code dL/dc}, returns {@code dL/da} treating {@code b} as held
     * fixed.
     */
    public static double[] mulBackwardA(double[] b, double[] dC) {
        double[] dA = new double[DIM];
        dA[0] =  dC[0]*b[0] + dC[1]*b[1] + dC[2]*b[2] + dC[3]*b[3];
        dA[1] = -dC[0]*b[1] + dC[1]*b[0] - dC[2]*b[3] + dC[3]*b[2];
        dA[2] =  dC[0]*b[2] - dC[1]*b[3] + dC[2]*b[0] - dC[3]*b[1];
        dA[3] =  dC[0]*b[3] + dC[1]*b[2] + dC[2]*b[1] + dC[3]*b[0];
        return dA;
    }

    /** Backward of {@code c = a · b} w.r.t. the right operand. */
    public static double[] mulBackwardB(double[] a, double[] dC) {
        double[] dB = new double[DIM];
        dB[0] =  dC[0]*a[0] + dC[1]*a[1] + dC[2]*a[2] + dC[3]*a[3];
        dB[1] = -dC[0]*a[1] + dC[1]*a[0] + dC[2]*a[3] - dC[3]*a[2];
        dB[2] =  dC[0]*a[2] + dC[1]*a[3] + dC[2]*a[0] + dC[3]*a[1];
        dB[3] =  dC[0]*a[3] - dC[1]*a[2] - dC[2]*a[1] + dC[3]*a[0];
        return dB;
    }

    /**
     * Conjugate sandwich: {@code y = q · v · q̄}. The split-quaternion
     * analog of a rotation by a unit quaternion — except split-quat
     * norms aren't positive-definite, so the sandwich rescales by
     * {@code N(q)} on the unit cells and stretches/compresses in the
     * hyperbolic directions.
     */
    public static double[] sandwich(double[] q, double[] v) {
        double[] r = mul(q, v);
        return mul(r, conj(q));
    }

    /**
     * Backward of {@code y = q · v · q̄}. Both occurrences of q
     * contribute to {@code dL/dq}; sum the two paths. Returns
     * {@code {dQ, dV}}.
     */
    public static double[][] sandwichBackward(double[] q, double[] v, double[] dY) {
        double[] qBar = conj(q);
        double[] r = mul(q, v);

        // y = r · qBar  ⇒  dL/dr from left-arg backward, dL/dqBar from right-arg backward.
        double[] dR = mulBackwardA(qBar, dY);
        double[] dQbar = mulBackwardB(r, dY);

        // r = q · v
        double[] dQfromR = mulBackwardA(v, dR);
        double[] dV = mulBackwardB(q, dR);

        // qBar = conj(q): scalar part identity, others negated.
        double[] dQfromQbar = new double[] { dQbar[0], -dQbar[1], -dQbar[2], -dQbar[3] };

        double[] dQ = new double[DIM];
        for (int i = 0; i < DIM; i++) dQ[i] = dQfromR[i] + dQfromQbar[i];
        return new double[][] { dQ, dV };
    }

    /** Split-quaternion norm form. Vanishes precisely on the null cone. */
    public static double norm(double[] a) {
        return a[0]*a[0] + a[1]*a[1] - a[2]*a[2] - a[3]*a[3];
    }
}
