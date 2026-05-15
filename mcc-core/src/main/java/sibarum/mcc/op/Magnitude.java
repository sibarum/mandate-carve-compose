package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: L2 norm. {@code u -> √(Σ u_i²)}. Returns a scalar.
 *
 * <p>Backward: {@code dL/du_i = gradOut · u_i / ‖u‖}. Zero-norm input
 * yields a zero gradient (the derivative is undefined at the origin;
 * returning zero is the standard subgradient).
 */
public final class Magnitude implements Differentiable {

    private double[] lastX;
    private double lastNorm;

    @Override
    public String name() {
        return "magnitude";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] a = ((MatrixValue) inputs.getFirst()).data();
        double s = 0.0;
        for (double v : a) {
            if (Double.isNaN(v)) {
                throw new IllegalArgumentException("Magnitude rejects NaN input");
            }
            s += v * v;
        }
        lastX = a.clone();
        lastNorm = Math.sqrt(s);
        return new NumberValue(lastNorm);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastX == null) throw new IllegalStateException("Magnitude backward without prior apply");
        double g = ((NumberValue) gradOutput).n();
        double[] dx = new double[lastX.length];
        if (lastNorm > 0.0) {
            for (int i = 0; i < lastX.length; i++) {
                dx[i] = g * lastX[i] / lastNorm;
            }
        }
        // else: zero-norm input → zero subgradient.
        return List.of(new MatrixValue(dx));
    }
}
