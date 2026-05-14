package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: unit-norm projection. {@code u -> u / ‖u‖}. Zero-norm
 * inputs short-circuit to the zero vector — keeps the substrate total
 * (no NaN from 0/0) without claiming a spurious unit direction.
 *
 * <p>The standard L2-normalize Jacobian
 * {@code ∂(u/‖u‖)/∂u = (I − ŷŷᵀ) / ‖u‖} is the gradient pattern any
 * training-time backward will need; this primitive is forward-only for
 * MVP. A future Differentiable extension will supply the analytic
 * backward.
 */
public final class Normalize implements Primitive {

    private static final double EPS = 1e-12;

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
            return new MatrixValue(out);
        }
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] / norm;
        }
        return new MatrixValue(out);
    }
}
