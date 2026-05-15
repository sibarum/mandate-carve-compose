package sibarum.strnn.demo;

import sibarum.strnn.hpb.HarmonicBasis;
import sibarum.strnn.hpb.PiecewisePolynomial;
import sibarum.strnn.hpb.SmoothedBasisElement;
import sibarum.strnn.hpb.SmoothedBasisElement.Kernel;

/**
 * T2 from {@code docs/harmonic_piecewise_basis.md}: verify that smoothing
 * preserves derivative pairing, {@code (tri_k * φ)' = sq_k * φ}.
 *
 * <p>The mathematical content reduces to two analytic identities on the
 * antiderivatives, which the demo verifies pointwise on a dense grid
 * (machine-precision exactness, no central-difference truncation error):
 * <ol>
 *   <li><b>Box pairing</b> ⇐ {@code F_sq ≡ tri} (i.e.,
 *       {@code sq_k.antiderivative() == tri_k}).
 *       Both sides of the pairing reduce to
 *       {@code (tri(x+w/2) − tri(x−w/2)) / w}.</li>
 *   <li><b>Tent pairing</b> ⇐ {@code G_sq ≡ F_tri} (i.e.,
 *       {@code sq_k.antiderivative().antiderivative()
 *              == tri_k.antiderivative()}). Both sides of the pairing
 *       then reduce to
 *       {@code (4/w²)·(F_tri(x+w/2) − 2 F_tri(x) + F_tri(x−w/2))}.</li>
 * </ol>
 *
 * <p>The demo also samples the {@link SmoothedBasisElement} class on a
 * coarse grid as a sanity check that the kernel evaluation matches the
 * analytic formulas (smoothed-tri vs an explicit closed-form
 * computation).
 *
 * <p>Falsifies F1 on the smoothed basis if any check fires.
 */
public final class HpbDerivativePairingDemo {

    private static final int[] FREQS = { 1, 2, 3, 4, 8 };
    private static final double[] WIDTHS_FRAC = { 0.125, 0.0625 };
    private static final int GRID_POINTS = 500;
    private static final double TOL = 1e-12;

    public static void main(String[] args) {
        System.out.println();
        System.out.println("=== HPB smoothed-basis derivative pairing (T2) ===");
        System.out.println("  exact pointwise checks on antiderivative identities");
        System.out.println("  tolerance " + TOL);
        System.out.println();

        int totalChecks = 0;
        int totalMismatch = 0;
        double maxAbsErr = 0.0;
        String worstWhere = "";

        System.out.println("--- Identity A: sq.antiderivative() == tri  (box pairing) ---");
        for (int k : FREQS) {
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq  = HarmonicBasis.sqK(k);
            PiecewisePolynomial F_sq = sq.antiderivative();
            double T = 1.0 / k;
            int mis = 0;
            double mx = 0.0;
            for (int i = 0; i < GRID_POINTS; i++) {
                double x = (i + 0.37) / GRID_POINTS * T;
                double a = F_sq.evaluate(x);
                double b = tri.evaluate(x);
                double err = Math.abs(a - b);
                if (err > TOL) mis++;
                if (err > mx) mx = err;
                if (err > maxAbsErr) {
                    maxAbsErr = err;
                    worstWhere = String.format("ident=A k=%d x=%.4f", k, x);
                }
                totalChecks++;
            }
            totalMismatch += mis;
            System.out.printf("  k=%d: checks=%d mismatches=%d maxErr=%.3e%n",
                    k, GRID_POINTS, mis, mx);
        }

        System.out.println();
        System.out.println("--- Identity B: sq.antideriv().antideriv() == tri.antideriv()  (tent pairing) ---");
        for (int k : FREQS) {
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq  = HarmonicBasis.sqK(k);
            PiecewisePolynomial G_sq = sq.antiderivative().antiderivative();
            PiecewisePolynomial F_tri = tri.antiderivative();
            double T = 1.0 / k;
            int mis = 0;
            double mx = 0.0;
            for (int i = 0; i < GRID_POINTS; i++) {
                double x = (i + 0.37) / GRID_POINTS * T;
                double a = G_sq.evaluate(x);
                double b = F_tri.evaluate(x);
                double err = Math.abs(a - b);
                if (err > TOL) mis++;
                if (err > mx) mx = err;
                if (err > maxAbsErr) {
                    maxAbsErr = err;
                    worstWhere = String.format("ident=B k=%d x=%.4f", k, x);
                }
                totalChecks++;
            }
            totalMismatch += mis;
            System.out.printf("  k=%d: checks=%d mismatches=%d maxErr=%.3e%n",
                    k, GRID_POINTS, mis, mx);
        }

        System.out.println();
        System.out.println("--- SmoothedBasisElement sanity: closed-form vs class output ---");
        for (Kernel kernel : new Kernel[] { Kernel.BOX, Kernel.TENT }) {
            for (double wFrac : WIDTHS_FRAC) {
                for (int k : FREQS) {
                    double T = 1.0 / k;
                    double w = T * wFrac;
                    PiecewisePolynomial tri = HarmonicBasis.triK(k);
                    PiecewisePolynomial F_tri = tri.antiderivative();
                    PiecewisePolynomial G_tri = F_tri.antiderivative();
                    SmoothedBasisElement smTri = (kernel == Kernel.BOX)
                            ? SmoothedBasisElement.box(tri, w)
                            : SmoothedBasisElement.tent(tri, w);

                    int mis = 0;
                    double mx = 0.0;
                    int N = 100;
                    for (int i = 0; i < N; i++) {
                        double x = (i + 0.37) / N * T;
                        double fromClass = smTri.evaluate(x);
                        double fromFormula = (kernel == Kernel.BOX)
                                ? (F_tri.evaluate(x + w / 2) - F_tri.evaluate(x - w / 2)) / w
                                : (4.0 / (w * w)) * (G_tri.evaluate(x + w / 2)
                                        - 2.0 * G_tri.evaluate(x)
                                        + G_tri.evaluate(x - w / 2));
                        double err = Math.abs(fromClass - fromFormula);
                        if (err > TOL) mis++;
                        if (err > mx) mx = err;
                        if (err > maxAbsErr) {
                            maxAbsErr = err;
                            worstWhere = String.format("class kernel=%s w=%.4f k=%d x=%.4f",
                                    kernel, w, k, x);
                        }
                        totalChecks++;
                    }
                    totalMismatch += mis;
                    System.out.printf("  %-4s w=%.4f k=%d: checks=%d mismatches=%d maxErr=%.3e%n",
                            kernel, w, k, N, mis, mx);
                }
            }
        }

        System.out.println();
        System.out.println("  total checks: " + totalChecks);
        System.out.println("  total mismatches: " + totalMismatch);
        System.out.printf("  worst err: %.3e   (%s)%n", maxAbsErr, worstWhere);
        boolean pass = totalMismatch == 0;
        System.out.println("  RESULT: " + (pass ? "PASS" : "FAIL"));
        if (!pass) System.exit(1);
    }
}
