package sibarum.mcc.op;

import sibarum.mcc.op.util.TotalArithmetic;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (u, v) -> ⟨u, v⟩ / (‖u‖·‖v‖)}, returning a scalar in
 * {@code [-1, +1]}. Zero-norm vectors short-circuit to {@code 0}.
 *
 * <p>Backward (derived from cos = d / (n_u·n_v) where d = ⟨u,v⟩):
 * <pre>
 *   dL/du = gradOut · ( v / (n_u·n_v) − d·u / (n_u³·n_v) )
 *   dL/dv = gradOut · ( u / (n_u·n_v) − d·v / (n_u·n_v³) )
 * </pre>
 * Zero-norm inputs produce a zero gradient (the cosine surface is
 * undefined there).
 */
public final class CosineSimilarity implements Differentiable {

    private double[] lastU;
    private double[] lastV;
    private double lastNu;
    private double lastNv;
    private double lastDot;

    @Override
    public String name() {
        return "cosine-similarity";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX, ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] a = ((MatrixValue) inputs.get(0)).data();
        double[] b = ((MatrixValue) inputs.get(1)).data();
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "CosineSimilarity dim mismatch: " + a.length + " vs " + b.length);
        }
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot = TotalArithmetic.totalAdd(dot, TotalArithmetic.totalMul(a[i], b[i]));
            na = TotalArithmetic.totalAdd(na, TotalArithmetic.totalMul(a[i], a[i]));
            nb = TotalArithmetic.totalAdd(nb, TotalArithmetic.totalMul(b[i], b[i]));
        }
        lastU = a.clone();
        lastV = b.clone();
        lastNu = Math.sqrt(na);
        lastNv = Math.sqrt(nb);
        lastDot = dot;
        if (lastNu == 0.0 || lastNv == 0.0) {
            return new NumberValue(0.0);
        }
        return new NumberValue(TotalArithmetic.totalDiv(dot, lastNu * lastNv));
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastU == null) throw new IllegalStateException("CosineSimilarity backward without prior apply");
        double g = ((NumberValue) gradOutput).n();
        double[] dU = new double[lastU.length];
        double[] dV = new double[lastV.length];
        if (lastNu > 0.0 && lastNv > 0.0) {
            double denom = lastNu * lastNv;
            double nuCubedNv = lastNu * lastNu * lastNu * lastNv;
            double nuNvCubed = lastNu * lastNv * lastNv * lastNv;
            for (int i = 0; i < lastU.length; i++) {
                dU[i] = g * (lastV[i] / denom - lastDot * lastU[i] / nuCubedNv);
                dV[i] = g * (lastU[i] / denom - lastDot * lastV[i] / nuNvCubed);
            }
        }
        return List.of(new MatrixValue(dU), new MatrixValue(dV));
    }
}
