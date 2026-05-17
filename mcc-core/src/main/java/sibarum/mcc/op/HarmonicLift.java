package sibarum.mcc.op;

import sibarum.mcc.op.advanced.HarmonicBasis;
import sibarum.mcc.op.advanced.PiecewisePolynomial;
import sibarum.mcc.op.advanced.SmoothedBasisElement;
import sibarum.mcc.op.advanced.SmoothedBasisElement.Kernel;
import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;
import java.util.Map;

/**
 * Per-dimension harmonic piecewise basis lift. Maps a {@code MATRIX} of
 * length {@code inputDim} to a {@code MATRIX} of length
 * {@code inputDim · 2K} by applying the same K-frequency lift to each
 * input component independently and concatenating.
 *
 * <p>For each input component {@code x_i}, the lift produces the
 * 2K-vector {@code [tri_1(x_i), sq_1(x_i), …, tri_K(x_i), sq_K(x_i)]}.
 * Basis elements are periodic with period {@code 1/k} (unit-period
 * convention) and amplitude 1; the linear readout downstream handles
 * scale.
 *
 * <p>Three kernel choices via {@link SmoothedBasisElement}:
 * <ul>
 *   <li><b>delta</b>: raw piecewise-linear basis (width irrelevant).</li>
 *   <li><b>box</b>: smoothing window of width {@code widthFrac · T_k}
 *       per frequency. Piecewise quadratic, C^0.</li>
 *   <li><b>tent</b>: same width, piecewise cubic, C^1.</li>
 * </ul>
 * Width is specified as a fraction of each basis element's period so the
 * smoothing is consistent across frequencies (a fixed absolute width
 * would over-smooth high-k elements and under-smooth low-k ones).
 *
 * <p>No learned parameters — the basis is structural. Composes naturally
 * with {@link Linear} for a linear readout, or with an {@link Mlp Block}
 * for a non-linear readout. For 2D-XOR-style cross-dimension tasks,
 * composing a single {@code HarmonicLift} with a downstream
 * non-linearity recovers cross-feature interactions that the lift
 * alone (linear in each input) cannot express.
 *
 * <p>See {@code docs/17-harmonic-piecewise-basis.md} for the iter-by-iter
 * empirical record. Promoted from {@code sibarum.strnn.hpb} after
 * surviving iters 1, 1.5, and 2 in the research line.
 */
public final class HarmonicLift implements Differentiable, Configurable {

    private final int K;
    private final int inputDim;
    private final Kernel kernel;
    private final double widthFrac;
    private final int featPerInput;
    private final int outDim;

    // [inputDim][K] arrays of smoothed-basis elements per frequency per input slot.
    // Each input slot uses the same K basis elements; we materialize per-slot copies
    // only because evaluate/derivative cache no state (the same instance is reusable),
    // so we share one set of K arrays across all slots.
    private final SmoothedBasisElement[] smTri;
    private final SmoothedBasisElement[] smSq;

    private double[] lastInput;

    public HarmonicLift(int K, int inputDim, Kernel kernel, double widthFrac) {
        if (K <= 0) throw new IllegalArgumentException("K must be positive: " + K);
        if (inputDim <= 0) throw new IllegalArgumentException("inputDim must be positive: " + inputDim);
        if (kernel == null) throw new IllegalArgumentException("kernel must not be null");
        if (kernel != Kernel.DELTA && widthFrac <= 0.0) {
            throw new IllegalArgumentException(
                    "widthFrac must be positive for non-delta kernel: " + widthFrac);
        }
        this.K = K;
        this.inputDim = inputDim;
        this.kernel = kernel;
        this.widthFrac = widthFrac;
        this.featPerInput = 2 * K;
        this.outDim = inputDim * featPerInput;
        this.smTri = new SmoothedBasisElement[K];
        this.smSq = new SmoothedBasisElement[K];
        for (int i = 0; i < K; i++) {
            int k = i + 1;
            double T = 1.0 / k;
            double w = T * widthFrac;
            PiecewisePolynomial tri = HarmonicBasis.triK(k);
            PiecewisePolynomial sq = HarmonicBasis.sqK(k);
            smTri[i] = wrap(kernel, tri, w);
            smSq[i] = wrap(kernel, sq, w);
        }
    }

    private static SmoothedBasisElement wrap(Kernel kernel, PiecewisePolynomial p, double w) {
        return switch (kernel) {
            case DELTA -> SmoothedBasisElement.delta(p);
            case BOX -> SmoothedBasisElement.box(p, w);
            case TENT -> SmoothedBasisElement.tent(p, w);
        };
    }

    public int K() { return K; }
    public int inputDim() { return inputDim; }
    public Kernel kernel() { return kernel; }
    public double widthFrac() { return widthFrac; }
    public int outDim() { return outDim; }

    @Override
    public String name() {
        return "harmonic-lift";
    }

    @Override
    public Map<String, Object> config() {
        return Map.of(
                "K", K,
                "inputDim", inputDim,
                "kernel", kernel.name(),
                "widthFrac", widthFrac
        );
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length != inputDim) {
            throw new IllegalArgumentException(
                    "HarmonicLift input length " + x.length + " != inputDim " + inputDim);
        }
        double[] out = new double[outDim];
        for (int i = 0; i < inputDim; i++) {
            double xi = x[i];
            if (Double.isNaN(xi)) {
                throw new IllegalArgumentException("HarmonicLift rejects NaN input");
            }
            int base = i * featPerInput;
            for (int j = 0; j < K; j++) {
                out[base + 2 * j]     = smTri[j].evaluate(xi);
                out[base + 2 * j + 1] = smSq[j].evaluate(xi);
            }
        }
        lastInput = x.clone();
        return new MatrixValue(out);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("HarmonicLift backward called without prior apply");
        }
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != outDim) {
            throw new IllegalArgumentException(
                    "HarmonicLift gradOutput length " + g.length + " != outDim " + outDim);
        }
        double[] dx = new double[inputDim];
        for (int i = 0; i < inputDim; i++) {
            double xi = lastInput[i];
            int base = i * featPerInput;
            double s = 0.0;
            for (int j = 0; j < K; j++) {
                s += g[base + 2 * j]     * smTri[j].evaluateDerivative(xi);
                s += g[base + 2 * j + 1] * smSq[j].evaluateDerivative(xi);
            }
            dx[i] = s;
        }
        return List.of(new MatrixValue(dx));
    }
}
