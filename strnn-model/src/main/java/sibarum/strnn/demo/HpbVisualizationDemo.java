package sibarum.strnn.demo;

import sibarum.strnn.hpb.HarmonicBasis;
import sibarum.strnn.hpb.PiecewisePolynomial;
import sibarum.strnn.hpb.SmoothedBasisElement;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * ASCII plots of the HPB basis. Run to see what the math actually looks like.
 */
public final class HpbVisualizationDemo {

    public static void main(String[] args) {
        PiecewisePolynomial tri1 = HarmonicBasis.triK(1);
        PiecewisePolynomial sq1  = HarmonicBasis.sqK(1);
        PiecewisePolynomial tri2 = HarmonicBasis.triK(2);
        PiecewisePolynomial sq2  = HarmonicBasis.sqK(2);

        System.out.println();
        System.out.println("============================================================");
        System.out.println(" 1.  Raw basis at K=1:  tri_1 (#)  and  sq_1 (*)");
        System.out.println("     period = 1, amplitude of tri = 1, amplitude of sq = 4k = 4");
        System.out.println("============================================================");
        AsciiPlot p1 = new AsciiPlot(78, 15, 0.0, 1.0, -4.5, 4.5);
        p1.axes();
        p1.curve(tri1::evaluate, '#');
        p1.curve(sq1::evaluate, '*');
        System.out.println(p1.render());

        System.out.println("============================================================");
        System.out.println(" 2.  Raw basis at K=2:  tri_2 (#)  and  sq_2 (*)");
        System.out.println("     period = 1/2, twice as fast; amplitude of sq = 4k = 8");
        System.out.println("============================================================");
        AsciiPlot p2 = new AsciiPlot(78, 15, 0.0, 1.0, -9.0, 9.0);
        p2.axes();
        p2.curve(tri2::evaluate, '#');
        p2.curve(sq2::evaluate, '*');
        System.out.println(p2.render());

        System.out.println("============================================================");
        System.out.println(" 3.  Smoothing changes sq_1's shape:  raw (.)  vs  smoothed (#)");
        System.out.println("     left to right: delta, box w=T/8, tent w=T/8, tent w=T/4");
        System.out.println("============================================================");

        SmoothedBasisElement smDelta = SmoothedBasisElement.delta(sq1);
        SmoothedBasisElement smBox   = SmoothedBasisElement.box(sq1, 1.0 / 8);
        SmoothedBasisElement smTent  = SmoothedBasisElement.tent(sq1, 1.0 / 8);
        SmoothedBasisElement smTent4 = SmoothedBasisElement.tent(sq1, 1.0 / 4);

        System.out.println();
        System.out.println("--- delta kernel: raw piecewise constant ---");
        AsciiPlot pd = new AsciiPlot(78, 11, 0.0, 1.0, -5.0, 5.0);
        pd.axes();
        pd.curve(smDelta::evaluate, '#');
        System.out.println(pd.render());

        System.out.println("--- box kernel, width = T/8 (= 0.125) ---");
        AsciiPlot pb = new AsciiPlot(78, 11, 0.0, 1.0, -5.0, 5.0);
        pb.axes();
        pb.curve(sq1::evaluate, '.');
        pb.curve(smBox::evaluate, '#');
        System.out.println(pb.render());

        System.out.println("--- tent kernel, width = T/8 ---");
        AsciiPlot pt = new AsciiPlot(78, 11, 0.0, 1.0, -5.0, 5.0);
        pt.axes();
        pt.curve(sq1::evaluate, '.');
        pt.curve(smTent::evaluate, '#');
        System.out.println(pt.render());

        System.out.println("--- tent kernel, width = T/4 (more aggressive) ---");
        AsciiPlot pt4 = new AsciiPlot(78, 11, 0.0, 1.0, -5.0, 5.0);
        pt4.axes();
        pt4.curve(sq1::evaluate, '.');
        pt4.curve(smTent4::evaluate, '#');
        System.out.println(pt4.render());

        System.out.println("============================================================");
        System.out.println(" 4.  Why K=1 solves 1D XOR:  XOR points (@) on sq_1 (*)");
        System.out.println("     x = (4n+1)/16, n=0..3, labels (0, 1, 1, 0).");
        System.out.println("     Each label-0 point lands on sq_1 = +4; each label-1 on -4.");
        System.out.println("     One linear weight w_sq = -1/4 reads off the XOR pattern.");
        System.out.println("============================================================");
        AsciiPlot p4 = new AsciiPlot(78, 11, 0.0, 1.0, -5.0, 5.0);
        p4.axes();
        p4.curve(sq1::evaluate, '*');
        double[] xorXs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        int[] xorLabels = { 0, 1, 1, 0 };
        for (int i = 0; i < xorXs.length; i++) {
            double x = xorXs[i];
            double y = sq1.evaluate(x);
            p4.point(x, y, '@');
            System.out.printf("    x = %5.4f   sq_1 = %+.1f   label = %d%n",
                    x, y, xorLabels[i]);
        }
        System.out.println();
        System.out.println(p4.render());

        System.out.println("============================================================");
        System.out.println(" 5.  T7 smooth-function approximation");
        System.out.println("     target = exp(sin(2*pi*x)) + sin(4*pi*x)*cos(6*pi*x)  (o)");
        System.out.println("     overlaid: delta K=16 fit (.)  and  box K=16 fit (#)");
        System.out.println("============================================================");
        AsciiPlot p5 = new AsciiPlot(78, 17, 0.0, 1.0, -1.5, 4.0);
        p5.axes();
        p5.curve(HpbVisualizationDemo::targetFn, 'o');
        p5.curve(x -> approximation(x, 16, false), '.');
        p5.curve(x -> approximation(x, 16, true), '#');
        System.out.println(p5.render());
    }

    private static double targetFn(double x) {
        return Math.exp(Math.sin(2 * Math.PI * x))
                + Math.sin(4 * Math.PI * x) * Math.cos(6 * Math.PI * x);
    }

    /** Returns the LS-best linear-readout prediction at x for K-frequency lift, optionally box-smoothed. */
    private static double approximation(double x, int K, boolean smoothed) {
        int N = 128;
        double[] xs = new double[N];
        double[] ys = new double[N];
        for (int i = 0; i < N; i++) {
            xs[i] = (i + 0.5) / N;
            ys[i] = targetFn(xs[i]);
        }
        SmoothedBasisElement[] smTri = new SmoothedBasisElement[K];
        SmoothedBasisElement[] smSq = new SmoothedBasisElement[K];
        for (int j = 0; j < K; j++) {
            int k = j + 1;
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq = HarmonicBasis.sqK(k);
            if (smoothed) {
                double w = 0.125 / k;
                smTri[j] = SmoothedBasisElement.box(tri, w);
                smSq[j] = SmoothedBasisElement.box(sq, w);
            } else {
                smTri[j] = SmoothedBasisElement.delta(tri);
                smSq[j] = SmoothedBasisElement.delta(sq);
            }
        }
        int F = 2 * K + 1;
        double[][] A = new double[N][F];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                A[i][2 * j] = smTri[j].evaluate(xs[i]);
                A[i][2 * j + 1] = smSq[j].evaluate(xs[i]);
            }
            A[i][F - 1] = 1.0;
        }
        double[] c = solveLeastSquares(A, ys);
        double pred = c[F - 1];
        for (int j = 0; j < K; j++) {
            pred += c[2 * j] * smTri[j].evaluate(x);
            pred += c[2 * j + 1] * smSq[j].evaluate(x);
        }
        return pred;
    }

    private static double[] solveLeastSquares(double[][] A, double[] b) {
        int n = A.length, p = A[0].length;
        double[][] AtA = new double[p][p];
        double[] Atb = new double[p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                Atb[j] += A[i][j] * b[i];
                for (int k = 0; k < p; k++) AtA[j][k] += A[i][j] * A[i][k];
            }
        }
        double[][] M = new double[p][p + 1];
        for (int i = 0; i < p; i++) {
            System.arraycopy(AtA[i], 0, M[i], 0, p);
            M[i][p] = Atb[i];
        }
        for (int piv = 0; piv < p; piv++) {
            int mx = piv;
            for (int r = piv + 1; r < p; r++) if (Math.abs(M[r][piv]) > Math.abs(M[mx][piv])) mx = r;
            if (mx != piv) { double[] t = M[piv]; M[piv] = M[mx]; M[mx] = t; }
            for (int r = piv + 1; r < p; r++) {
                double f = M[r][piv] / M[piv][piv];
                for (int c = piv; c <= p; c++) M[r][c] -= f * M[piv][c];
            }
        }
        double[] x = new double[p];
        for (int i = p - 1; i >= 0; i--) {
            double s = M[i][p];
            for (int j = i + 1; j < p; j++) s -= M[i][j] * x[j];
            x[i] = s / M[i][i];
        }
        return x;
    }

    static final class AsciiPlot {
        final int w, h;
        final double xMin, xMax, yMin, yMax;
        final char[][] g;

        AsciiPlot(int w, int h, double xMin, double xMax, double yMin, double yMax) {
            this.w = w; this.h = h;
            this.xMin = xMin; this.xMax = xMax;
            this.yMin = yMin; this.yMax = yMax;
            this.g = new char[h][w];
            for (char[] r : g) Arrays.fill(r, ' ');
        }

        int xToCol(double x) {
            return (int) Math.round((x - xMin) / (xMax - xMin) * (w - 1));
        }

        int yToRow(double y) {
            return (int) Math.round((yMax - y) / (yMax - yMin) * (h - 1));
        }

        void axes() {
            int r0 = yToRow(0);
            if (r0 >= 0 && r0 < h) for (int c = 0; c < w; c++) g[r0][c] = '-';
            int c0 = xToCol(0);
            if (c0 >= 0 && c0 < w) for (int r = 0; r < h; r++) if (g[r][c0] == ' ') g[r][c0] = '|';
            int rT = yToRow(1.0);
            if (rT >= 0 && rT < h) g[rT][0] = '1';
            int rB = yToRow(-1.0);
            if (rB >= 0 && rB < h) g[rB][0] = '-';
        }

        void curve(DoubleUnaryOperator f, char marker) {
            for (int c = 0; c < w; c++) {
                double x = xMin + (xMax - xMin) * c / (w - 1);
                double y = f.applyAsDouble(x);
                int r = yToRow(y);
                if (r >= 0 && r < h) g[r][c] = marker;
            }
        }

        void point(double x, double y, char marker) {
            int c = xToCol(x), r = yToRow(y);
            if (c >= 0 && c < w && r >= 0 && r < h) g[r][c] = marker;
        }

        String render() {
            StringBuilder sb = new StringBuilder();
            for (char[] r : g) sb.append(new String(r)).append('\n');
            return sb.toString();
        }
    }
}
