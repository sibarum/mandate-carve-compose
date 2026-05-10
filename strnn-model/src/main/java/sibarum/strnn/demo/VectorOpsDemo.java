package sibarum.strnn.demo;

import sibarum.strnn.cache.SimilarityGate;
import sibarum.strnn.cache.VectorAdd;
import sibarum.strnn.cache.VectorMul;
import sibarum.strnn.cache.VectorSub;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Vector ops layer: VectorTransform (learnable), VectorAdd / Sub / Mul,
 * SimilarityGate. All built on the {@code TotalArithmetic} floor — every
 * componentwise operation is total, so ±∞ inputs are handled without NaN.
 *
 * Diagnostics:
 *   1) VectorTransform learns to be the identity (smallest test of the
 *      learnable-linear-transform plumbing).
 *   2) Binary ops produce expected values on finite inputs and behave
 *      correctly under ±∞ edge cases.
 *   3) SimilarityGate gates aligned / orthogonal / opposed query-ref pairs
 *      with the right scale and sign.
 *   4) All four primitives chain together in a single forward pipeline:
 *      transform → sub → mul → gate. No NaN, no exceptions.
 */
public final class VectorOpsDemo {

    private static final double EPS = 1e-6;

    public static void main(String[] args) {
        runVectorTransform();
        System.out.println();
        runBinaryOps();
        System.out.println();
        runSimilarityGate();
        System.out.println();
        runComposition();
        System.out.println("\nVector ops layer: total componentwise arithmetic + "
                + "learnable Wv + similarity gating, all composing cleanly.");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 1: VectorTransform converges to identity.
    // ---------------------------------------------------------------------

    private static void runVectorTransform() {
        System.out.println("=== Diagnostic 1: VectorTransform learns identity (Wv → v) ===");
        int dim = 6;
        VectorTransform t = new VectorTransform(dim, dim, 99L);

        // Training: sample random vectors, target = input. Gradient pushes W → I.
        java.util.Random rng = new java.util.Random(0L);
        double lr = 0.05;
        int epochs = 800;
        int batch = 8;
        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < batch; s++) {
                double[] v = new double[dim];
                for (int i = 0; i < dim; i++) v[i] = rng.nextGaussian();
                MatrixValue mv = new MatrixValue(v);
                t.apply(List.of(mv));
                t.backward(mv);
                t.step(lr);
            }
        }

        // Verify: identity behavior on a held-out sample.
        java.util.Random testRng = new java.util.Random(123L);
        int passes = 0;
        int trials = 16;
        for (int trial = 0; trial < trials; trial++) {
            double[] v = new double[dim];
            for (int i = 0; i < dim; i++) v[i] = testRng.nextGaussian();
            MatrixValue out = (MatrixValue) t.apply(List.of(new MatrixValue(v)));
            double err = l2(sub(out.data(), v));
            if (err < 0.05) passes++;
        }
        System.out.printf(Locale.ROOT, "  identity recovery: %d / %d trials within err < 0.05%n",
                passes, trials);
        require(passes == trials, "VectorTransform did not converge to identity");
        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 2: binary ops on finite + edge-case inputs.
    // ---------------------------------------------------------------------

    private static void runBinaryOps() {
        System.out.println("=== Diagnostic 2: VectorAdd / Sub / Mul, including ±∞ ===");
        VectorAdd add = new VectorAdd();
        VectorSub sub = new VectorSub();
        VectorMul mul = new VectorMul();

        double posInf = Double.POSITIVE_INFINITY;
        double negInf = Double.NEGATIVE_INFINITY;

        // Finite cases.
        verify(add, vec(1, 2, 3), vec(4, 5, 6), vec(5, 7, 9), "add finite");
        verify(sub, vec(4, 5, 6), vec(1, 2, 3), vec(3, 3, 3), "sub finite");
        verify(mul, vec(2, 3, 4), vec(5, 6, 7), vec(10, 18, 28), "mul finite (Hadamard)");

        // Edge cases — totality at vector level.
        verify(add, vec(posInf, 1, 0), vec(negInf, 2, 0), vec(0, 3, 0), "add ±∞ in component 0");
        verify(sub, vec(posInf, 5, 0), vec(posInf, 3, 0), vec(0, 2, 0), "sub same-sign ∞ in component 0");
        verify(mul, vec(0, 2, posInf), vec(posInf, 3, 0), vec(1, 6, 1), "mul 0·∞ in components 0 and 2");
        verify(mul, vec(0, 1), vec(negInf, 1), vec(-1, 1), "mul 0·−∞ → −1 (sign-aware)");

        System.out.println("  PASS (7 vector cases over the total-arithmetic floor)");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 3: SimilarityGate gates correctly.
    // ---------------------------------------------------------------------

    private static void runSimilarityGate() {
        System.out.println("=== Diagnostic 3: SimilarityGate (aligned / orthogonal / opposed) ===");
        SimilarityGate gate = new SimilarityGate();

        // Aligned: q · q → cos = 1, output = q.
        double[] q = {1, 2, 3, 4};
        Value out = gate.apply(List.of(new MatrixValue(q), new MatrixValue(q)));
        double[] r = ((MatrixValue) out).data();
        require(approxEqual(r, q), "aligned gate did not pass through: " + Arrays.toString(r));
        System.out.printf(Locale.ROOT, "  cos(q, q) = 1   →   gated = q       (component-0: %+.3f)%n", r[0]);

        // Orthogonal: q vs perp → cos ≈ 0, output ≈ zero.
        double[] perp = {-2, 1, -4, 3};   // dot with q = -2 + 2 - 12 + 12 = 0
        out = gate.apply(List.of(new MatrixValue(q), new MatrixValue(perp)));
        r = ((MatrixValue) out).data();
        require(l2(r) < EPS, "orthogonal gate did not zero: " + Arrays.toString(r));
        System.out.printf(Locale.ROOT, "  cos(q, q⊥) = 0  →   gated = 0       (norm: %+.3f)%n", l2(r));

        // Opposed: q vs −q → cos = −1, output = −q.
        double[] negQ = sub(new double[q.length], q);
        out = gate.apply(List.of(new MatrixValue(q), new MatrixValue(negQ)));
        r = ((MatrixValue) out).data();
        require(approxEqual(r, negQ), "opposed gate did not flip: " + Arrays.toString(r));
        System.out.printf(Locale.ROOT, "  cos(q, −q) = -1 →   gated = −q      (component-0: %+.3f)%n", r[0]);

        // Zero ref: degenerate case, gate = 0.
        double[] zero = new double[q.length];
        out = gate.apply(List.of(new MatrixValue(q), new MatrixValue(zero)));
        r = ((MatrixValue) out).data();
        require(l2(r) < EPS, "zero-ref did not gate to zero: " + Arrays.toString(r));
        System.out.println("  zero ref         →   gated = 0       (no spurious all-pass)");

        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Diagnostic 4: composition.
    // ---------------------------------------------------------------------

    private static void runComposition() {
        System.out.println("=== Diagnostic 4: composition (transform → sub → mul → gate) ===");
        int dim = 4;
        VectorTransform t = new VectorTransform(dim, dim, 7L);
        VectorAdd add = new VectorAdd();
        VectorSub sub = new VectorSub();
        VectorMul mul = new VectorMul();
        SimilarityGate gate = new SimilarityGate();

        MatrixValue u = new MatrixValue(new double[]{1, 0, -1, 2});
        MatrixValue v = new MatrixValue(new double[]{0, 1, 1, -1});
        MatrixValue ref = new MatrixValue(new double[]{1, 1, 0, 0});

        Value tu = t.apply(List.of(u));
        Value diff = sub.apply(List.of(tu, v));
        Value sum = add.apply(List.of(diff, v));     // diff + v should ≈ tu
        Value scaled = mul.apply(List.of(sum, u));
        Value gated = gate.apply(List.of(scaled, ref));

        double[] r = ((MatrixValue) gated).data();
        require(r.length == dim, "wrong output dim from composition");
        for (double x : r) {
            require(!Double.isNaN(x), "NaN escaped composition");
        }
        System.out.printf(Locale.ROOT,
                "  pipeline ran clean. final vector norm: %.3f, no NaN.%n", l2(r));
        System.out.println("  PASS");
    }

    // ---------------------------------------------------------------------
    // Helpers.
    // ---------------------------------------------------------------------

    private static double[] vec(double... v) {
        return v;
    }

    private static void verify(sibarum.strnn.primitive.Primitive p,
                               double[] a, double[] b, double[] expected, String label) {
        Value out = p.apply(List.of(new MatrixValue(a), new MatrixValue(b)));
        double[] r = ((MatrixValue) out).data();
        require(r.length == expected.length, "[" + label + "] dim mismatch");
        for (int i = 0; i < r.length; i++) {
            require(componentEquals(r[i], expected[i]),
                    "[" + label + "] component " + i + ": expected " + expected[i] + ", got " + r[i]);
        }
        for (double x : r) {
            require(!Double.isNaN(x), "[" + label + "] NaN escaped");
        }
        System.out.printf(Locale.ROOT, "  %-45s → %s%n", label, Arrays.toString(r));
    }

    private static boolean componentEquals(double a, double b) {
        if (Double.isInfinite(a) && Double.isInfinite(b)) {
            return Math.signum(a) == Math.signum(b);
        }
        return Math.abs(a - b) < EPS;
    }

    private static boolean approxEqual(double[] a, double[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > EPS) return false;
        }
        return true;
    }

    private static double[] sub(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
        return r;
    }

    private static double l2(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    private static void require(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }
}
