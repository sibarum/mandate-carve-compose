package sibarum.strnn.ksqp;

import java.util.ArrayList;
import java.util.List;

/**
 * Polynomial (Π-net) lift: maps a vector k ∈ ℝⁿ to its vector of all
 * degree-d monomials. For (n=4, d=2) that's 10 monomials including
 * cross-terms (k_0 k_1, k_0 k_2, …); for general n and d the count is
 * C(n + d − 1, d) (multiset coefficient).
 *
 * <p>The lift itself is parameter-free; a separate learned projection
 * P_d : ℝ^{C(n+d-1, d)} → ℝ⁴ takes the monomial vector down to four
 * dimensions for the split-quaternion sandwich. See {@link KsqpModel}
 * for where P_d lives.
 *
 * <p>Convention for monomial enumeration: lex order on the multi-index
 * (α_0, α_1, …, α_{n-1}) with Σα_i = d. Highest α_0 first. Reproducible
 * across runs, so the gradient-trained P_d matches the same row order
 * every time.
 */
public final class PolyLift {

    private PolyLift() {}

    /** Multiset coefficient C(n + d - 1, d). */
    public static int monomialCount(int n, int d) {
        if (d == 0) return 1;
        if (n == 0) return 0;
        long num = 1;
        long den = 1;
        for (int i = 1; i <= d; i++) {
            num *= (n + d - i);
            den *= i;
        }
        return (int) (num / den);
    }

    /**
     * Enumerate all multi-indices (α_0, …, α_{n-1}) of length n summing
     * to d. Each returned array is fresh. Lex order, α_0 first.
     */
    public static int[][] enumerateMonomials(int n, int d) {
        List<int[]> out = new ArrayList<>(monomialCount(n, d));
        int[] alpha = new int[n];
        if (n == 0) {
            if (d == 0) out.add(alpha);
            return out.toArray(new int[0][]);
        }
        enumRec(alpha, 0, d, out);
        return out.toArray(new int[0][]);
    }

    private static void enumRec(int[] alpha, int pos, int remaining, List<int[]> out) {
        int n = alpha.length;
        if (pos == n - 1) {
            alpha[pos] = remaining;
            out.add(alpha.clone());
            return;
        }
        for (int a = remaining; a >= 0; a--) {
            alpha[pos] = a;
            enumRec(alpha, pos + 1, remaining - a, out);
        }
    }

    /** Evaluate every monomial in the list at k. Output length = monomials.length. */
    public static double[] lift(double[] k, int[][] monomials) {
        double[] m = new double[monomials.length];
        for (int idx = 0; idx < monomials.length; idx++) {
            int[] alpha = monomials[idx];
            double v = 1.0;
            for (int j = 0; j < alpha.length; j++) {
                if (alpha[j] > 0) v *= intPow(k[j], alpha[j]);
            }
            m[idx] = v;
        }
        return m;
    }

    /**
     * Backward through the lift. Given dL/dm (gradient on the monomial
     * vector) and the input k at which the lift was evaluated, returns
     * dL/dk. Used by the gradient check; in the training loop k is
     * frozen so this isn't called.
     *
     * <p>∂M_α/∂k_j = α_j · k_j^{α_j - 1} · ∏_{l ≠ j} k_l^{α_l}, with the
     * α_j = 0 case yielding 0.
     */
    public static double[] liftBackward(double[] k, int[][] monomials, double[] dM) {
        double[] dK = new double[k.length];
        for (int idx = 0; idx < monomials.length; idx++) {
            int[] alpha = monomials[idx];
            for (int j = 0; j < k.length; j++) {
                if (alpha[j] == 0) continue;
                double partial = alpha[j];
                for (int l = 0; l < k.length; l++) {
                    if (l == j) {
                        if (alpha[j] >= 2) partial *= intPow(k[j], alpha[j] - 1);
                    } else if (alpha[l] > 0) {
                        partial *= intPow(k[l], alpha[l]);
                    }
                }
                dK[j] += dM[idx] * partial;
            }
        }
        return dK;
    }

    private static double intPow(double base, int exp) {
        if (exp == 0) return 1.0;
        if (exp == 1) return base;
        if (exp == 2) return base * base;
        double result = 1.0;
        double b = base;
        int e = exp;
        while (e > 0) {
            if ((e & 1) == 1) result *= b;
            e >>= 1;
            if (e > 0) b *= b;
        }
        return result;
    }
}
