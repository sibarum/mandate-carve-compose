package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (MatrixValue, MatrixValue) -> NumberValue} cosine similarity.
 *
 * Returns the scalar in {@code [-1, +1]}. Zero-norm vectors short-circuit to
 * {@code 0} (no information available — keeps the substrate total without
 * spurious "everything matches everything" semantics).
 *
 * The complementary primitive to {@link SimilarityGate}: where the gate emits
 * the gated vector {@code sim · q}, this emits the scalar {@code sim} itself,
 * useful for routing decisions that need to be compared, summed, or fed into
 * scalar-typed downstream nodes.
 */
public final class CosineSimilarity implements Primitive {

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
        if (na == 0.0 || nb == 0.0) {
            return new NumberValue(0.0);
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return new NumberValue(TotalArithmetic.totalDiv(dot, denom));
    }
}
