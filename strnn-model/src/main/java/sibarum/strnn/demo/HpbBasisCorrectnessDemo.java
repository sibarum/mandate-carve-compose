package sibarum.strnn.demo;

import sibarum.strnn.hpb.HarmonicBasis;
import sibarum.strnn.hpb.PiecewisePolynomial;

/**
 * T1 from {@code docs/harmonic_piecewise_basis.md}: verifies the harmonic
 * piecewise basis matches the §2.1 specification at known points, that
 * tri_k.derivative() equals sq_k coefficient-by-coefficient, and that the
 * derivative pairing holds pointwise on a dense grid away from breakpoints.
 *
 * <p>Falsifies F1 (implementation bug in the basis) if any check fires.
 * The other tests (T3 gradient check, T5 XOR) are downstream of this.
 */
public final class HpbBasisCorrectnessDemo {

    private static final double COEF_TOL = 1e-12;
    private static final double POINT_TOL = 1e-9;

    public static void main(String[] args) {
        int[] freqs = { 1, 2, 3 };
        int totalChecks = 0;
        int totalMismatches = 0;

        for (int k : freqs) {
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq = HarmonicBasis.sqK(k);
            PiecewisePolynomial dTri = tri.derivative();

            int pieceMismatches = 0;
            for (int p = 0; p < tri.numPieces(); p++) {
                double[] dCoef = dTri.coefficientsOf(p);
                double[] sCoef = sq.coefficientsOf(p);
                if (dCoef.length != sCoef.length) {
                    pieceMismatches++;
                    System.out.printf("    k=%d piece %d: coef-length mismatch%n", k, p);
                    continue;
                }
                for (int d = 0; d < dCoef.length; d++) {
                    if (Math.abs(dCoef[d] - sCoef[d]) > COEF_TOL) {
                        pieceMismatches++;
                        System.out.printf("    k=%d piece %d coef %d: tri'=%.6e  sq=%.6e%n",
                                k, p, d, dCoef[d], sCoef[d]);
                    }
                }
            }
            totalChecks += tri.numPieces();
            totalMismatches += pieceMismatches;
            System.out.printf("  k=%d: pieces=%d, coefficient mismatches (d/dx tri = sq)=%d%n",
                    k, tri.numPieces(), pieceMismatches);

            double T = 1.0 / k;
            int pointMismatches = 0;
            pointMismatches += checkValue("tri_" + k + "(0)",        tri.evaluate(0.0),        0.0);
            pointMismatches += checkValue("tri_" + k + "(T/4)",      tri.evaluate(T / 4),      1.0);
            pointMismatches += checkValue("tri_" + k + "(T/2)",      tri.evaluate(T / 2),      0.0);
            pointMismatches += checkValue("tri_" + k + "(3T/4)",     tri.evaluate(3 * T / 4), -1.0);
            pointMismatches += checkValue("tri_" + k + " periodic",
                    tri.evaluate(T + T / 8), tri.evaluate(T / 8));

            pointMismatches += checkValue("sq_" + k + "(T/8)",       sq.evaluate(T / 8),       4.0 * k);
            pointMismatches += checkValue("sq_" + k + "(3T/8)",      sq.evaluate(3 * T / 8),  -4.0 * k);
            pointMismatches += checkValue("sq_" + k + "(5T/8)",      sq.evaluate(5 * T / 8),  -4.0 * k);
            pointMismatches += checkValue("sq_" + k + "(7T/8)",      sq.evaluate(7 * T / 8),   4.0 * k);
            totalChecks += 9;
            totalMismatches += pointMismatches;

            int gridChecks = 0;
            int gridMismatches = 0;
            int N = 1000;
            for (int i = 0; i < N; i++) {
                double frac = (i + 0.37) / N;
                double x = frac * T;
                double d = dTri.evaluate(x);
                double s = sq.evaluate(x);
                gridChecks++;
                if (Math.abs(d - s) > COEF_TOL) gridMismatches++;
            }
            totalChecks += gridChecks;
            totalMismatches += gridMismatches;
            System.out.printf("  k=%d: dense-grid (away-from-breakpoints) checks=%d, mismatches=%d%n",
                    k, gridChecks, gridMismatches);
        }

        System.out.println();
        System.out.println("=== HPB basis correctness (T1) ===");
        System.out.println("  total checks: " + totalChecks);
        System.out.println("  total mismatches: " + totalMismatches);
        boolean pass = totalMismatches == 0;
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        if (!pass) System.exit(1);
    }

    private static int checkValue(String label, double got, double expected) {
        if (Math.abs(got - expected) > POINT_TOL) {
            System.out.printf("    MISMATCH %s: got=%.6f, expected=%.6f%n", label, got, expected);
            return 1;
        }
        return 0;
    }
}
