package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (query, ref) -> cos(query, ref) · query}. Soft
 * content-conditional gating. Inputs aligned (positive cosine) pass
 * through with positive scale; orthogonal inputs zero out; opposed
 * inputs pass through negated. The conditional behaviour <em>is</em>
 * the nonlinearity — no explicit activation is required for non-linear
 * routing.
 *
 * <p>Zero-norm vectors short-circuit to gate = 0 (no information
 * available), rather than letting cosine become 0/0.
 */
public final class SimilarityGate implements Primitive {

    @Override
    public String name() {
        return "similarity-gate";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] q = ((MatrixValue) inputs.get(0)).data();
        double[] r = ((MatrixValue) inputs.get(1)).data();
        if (q.length != r.length) {
            throw new IllegalArgumentException(
                    "SimilarityGate dim mismatch: " + q.length + " vs " + r.length);
        }
        double sim = cosine(q, r);
        double[] out = new double[q.length];
        for (int i = 0; i < q.length; i++) {
            out[i] = TotalArithmetic.totalMul(sim, q[i]);
        }
        return new MatrixValue(out);
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot = TotalArithmetic.totalAdd(dot, TotalArithmetic.totalMul(a[i], b[i]));
            na = TotalArithmetic.totalAdd(na, TotalArithmetic.totalMul(a[i], a[i]));
            nb = TotalArithmetic.totalAdd(nb, TotalArithmetic.totalMul(b[i], b[i]));
        }
        if (na == 0.0 || nb == 0.0) return 0.0;
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return TotalArithmetic.totalDiv(dot, denom);
    }
}
