package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: unit-norm projection. {@code u -> u / ‖u‖}. Zero-norm
 * input short-circuits to the zero vector and produces a zero
 * gradient (the unit-norm Jacobian is undefined at the origin).
 *
 * <p>Backward: standard L2-normalize Jacobian
 * {@code (I − ŷŷᵀ)/‖u‖}. Concretely,
 * {@code dL/du_j = (gradOut_j − ŷ_j · ⟨ŷ, gradOut⟩) / ‖u‖}.
 */
public final class Normalize implements Differentiable {

    private static final double EPS = 1e-12;

    private double[] lastUnit;
    private double lastNorm;

    @Override
    public String name() {
        return "normalize";
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
        double[] a = ((MatrixValue) inputs.getFirst()).data();
        double s = 0.0;
        for (double v : a) {
            if (Double.isNaN(v)) {
                throw new IllegalArgumentException("Normalize rejects NaN input");
            }
            s += v * v;
        }
        double norm = Math.sqrt(s);
        double[] out = new double[a.length];
        if (norm <= EPS) {
            lastUnit = out.clone();
            lastNorm = 0.0;
            return new MatrixValue(out);
        }
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] / norm;
        }
        lastUnit = out.clone();
        lastNorm = norm;
        return new MatrixValue(out);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastUnit == null) throw new IllegalStateException("Normalize backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        double[] dx = new double[g.length];
        if (lastNorm <= EPS) {
            return List.of(new MatrixValue(dx));
        }
        double dot = 0.0;
        for (int i = 0; i < g.length; i++) dot += lastUnit[i] * g[i];
        for (int j = 0; j < g.length; j++) {
            dx[j] = (g[j] - lastUnit[j] * dot) / lastNorm;
        }
        return List.of(new MatrixValue(dx));
    }
}
