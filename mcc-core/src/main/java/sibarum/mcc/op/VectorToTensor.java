package sibarum.mcc.op;

import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Reshape: {@code MatrixValue → TensorValue} with a configured shape.
 * The input must be a flat vector whose length matches the shape's
 * volume.
 *
 * <p>Backward: pass {@code gradOut.data()} straight back as a
 * {@link MatrixValue} (the inverse flatten).
 */
public final class VectorToTensor implements Differentiable, Configurable {

    private final int[] targetShape;
    private final int expectedLength;

    public VectorToTensor(int[] targetShape) {
        this.targetShape = targetShape.clone();
        this.expectedLength = TensorValue.volume(this.targetShape);
    }

    public int[] targetShape() { return targetShape.clone(); }

    @Override
    public String name() {
        return "vector-to-tensor";
    }

    @Override
    public Map<String, Object> config() {
        List<Integer> shape = new ArrayList<>(targetShape.length);
        for (int s : targetShape) shape.add(s);
        return Map.of("shape", shape);
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TENSOR;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double[] x = ((MatrixValue) inputs.getFirst()).data();
        if (x.length != expectedLength) {
            throw new IllegalArgumentException(
                    "VectorToTensor expects length " + expectedLength + " for shape "
                            + Arrays.toString(targetShape) + ", got " + x.length);
        }
        return new TensorValue(targetShape, x);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        TensorValue g = (TensorValue) gradOutput;
        if (!Arrays.equals(g.shape(), targetShape)) {
            throw new IllegalArgumentException(
                    "VectorToTensor gradOutput shape " + Arrays.toString(g.shape())
                            + " != " + Arrays.toString(targetShape));
        }
        return List.of(new MatrixValue(g.data()));
    }
}
