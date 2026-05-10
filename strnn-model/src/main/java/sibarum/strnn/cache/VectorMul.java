package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: componentwise (Hadamard) vector multiplication.
 * {@code (u, v) -> u ⊙ v}.
 *
 * Routed through {@link TotalArithmetic#totalMul} so {@code 0 · ±∞ = ±1}
 * (sign-preserving) and no NaN can leak. This is the gating-flavoured
 * binary product; vector addition is the superposition-flavoured one.
 */
public final class VectorMul implements Primitive {

    @Override
    public String name() {
        return "vector-mul";
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
        double[] a = ((MatrixValue) inputs.get(0)).data();
        double[] b = ((MatrixValue) inputs.get(1)).data();
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "VectorMul dim mismatch: " + a.length + " vs " + b.length);
        }
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            r[i] = TotalArithmetic.totalMul(a[i], b[i]);
        }
        return new MatrixValue(r);
    }
}
