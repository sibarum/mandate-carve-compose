package sibarum.mcc.op;

import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.QuaternionValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * No-input, output-only Trainable primitive: holds a learnable
 * {@link Value} of a configured type, returns it on every {@link #apply},
 * accumulates gradients on {@link #backward}, and applies SGD on
 * {@link #step}.
 *
 * <p>This is the GUI palette's "parameter" node — drag in a vector /
 * matrix / ternion / quaternion / tensor parameter and wire its output
 * anywhere the corresponding {@link ValueType} is expected. Trainable
 * weights inside blocks (Linear, Mlp) are still owned by those blocks;
 * {@code Parameter} is for free-standing learnable values.
 *
 * <p>Internal storage is always a flat {@code double[]}; the shape
 * field tells us how to package it into the appropriate Value subtype
 * (and is also a structural config for the registry).
 */
public final class Parameter implements Trainable, Differentiable, Parameterized, Configurable {

    private final ValueType outputType;
    private final int[] shape;
    private final double[] data;
    private double[] pendingGrad;

    /** Construct a Parameter with random Xavier-style initialization. */
    public Parameter(ValueType outputType, int[] shape, long seed) {
        this(outputType, shape, randomInit(shape, seed));
    }

    /** Construct a Parameter with an explicit initial value (used by the importer). */
    public Parameter(ValueType outputType, int[] shape, double[] initialData) {
        validateShape(outputType, shape);
        int expected = TensorValue.volume(shape);
        if (initialData.length != expected) {
            throw new IllegalArgumentException(
                    "Parameter data length " + initialData.length + " != shape volume " + expected
                            + " for shape " + Arrays.toString(shape));
        }
        this.outputType = outputType;
        this.shape = shape.clone();
        this.data = initialData.clone();
    }

    @Override
    public void reinitialize(long seed) {
        double[] fresh = randomInit(shape, seed);
        System.arraycopy(fresh, 0, data, 0, data.length);
        pendingGrad = null;
    }

    private static double[] randomInit(int[] shape, long seed) {
        int n = TensorValue.volume(shape);
        double bound = Math.sqrt(3.0 / Math.max(1, n));
        Random rng = new Random(seed);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = (rng.nextDouble() * 2.0 - 1.0) * bound;
        }
        return out;
    }

    private static void validateShape(ValueType type, int[] shape) {
        switch (type) {
            case NUMBER -> {
                if (shape.length != 0 && !(shape.length == 1 && shape[0] == 1)) {
                    throw new IllegalArgumentException(
                            "NUMBER Parameter must have shape [] or [1], got " + Arrays.toString(shape));
                }
            }
            case MATRIX -> {
                if (shape.length != 1) {
                    throw new IllegalArgumentException(
                            "MATRIX Parameter must have rank-1 shape [dim], got " + Arrays.toString(shape));
                }
            }
            case TERNION -> {
                if (shape.length != 1 || shape[0] != 3) {
                    throw new IllegalArgumentException(
                            "TERNION Parameter must have shape [3], got " + Arrays.toString(shape));
                }
            }
            case QUATERNION -> {
                if (shape.length != 1 || shape[0] != 4) {
                    throw new IllegalArgumentException(
                            "QUATERNION Parameter must have shape [4], got " + Arrays.toString(shape));
                }
            }
            case TENSOR -> {
                // Any non-negative shape is OK.
            }
            default -> throw new IllegalArgumentException(
                    "Parameter does not support output type " + type);
        }
    }

    public ValueType type() { return outputType; }
    public int[] shape() { return shape.clone(); }
    public double[] rawData() { return data; }  // direct internal; treat as immutable from outside

    @Override
    public String name() {
        return "parameter";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of();
    }

    @Override
    public ValueType outputType() {
        return outputType;
    }

    @Override
    public Value apply(List<Value> inputs) {
        if (!inputs.isEmpty()) {
            throw new IllegalArgumentException("Parameter takes no inputs");
        }
        return packageOutput();
    }

    private Value packageOutput() {
        return switch (outputType) {
            case NUMBER -> new NumberValue(data[0]);
            case MATRIX -> new MatrixValue(data);
            case TERNION -> new TernionValue(data);
            case QUATERNION -> new QuaternionValue(data);
            case TENSOR -> new TensorValue(shape, data);
            default -> throw new IllegalStateException("unsupported Parameter type " + outputType);
        };
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        double[] g = extractFlat(gradOutput);
        if (g.length != data.length) {
            throw new IllegalArgumentException(
                    "Parameter gradOutput dim " + g.length + " != " + data.length);
        }
        if (pendingGrad == null) pendingGrad = new double[data.length];
        for (int i = 0; i < data.length; i++) pendingGrad[i] += g[i];
        return List.of();  // no input slots
    }

    private double[] extractFlat(Value v) {
        return switch (outputType) {
            case NUMBER -> new double[] { ((NumberValue) v).n() };
            case MATRIX -> ((MatrixValue) v).data();
            case TERNION -> ((TernionValue) v).toArray();
            case QUATERNION -> ((QuaternionValue) v).toArray();
            case TENSOR -> ((TensorValue) v).data();
            default -> throw new IllegalStateException();
        };
    }

    @Override
    public void step(double lr) {
        if (pendingGrad == null) return;
        for (int i = 0; i < data.length; i++) {
            data[i] -= lr * pendingGrad[i];
        }
        pendingGrad = null;
    }

    @Override
    public Object trainableIdentity() {
        return this;
    }

    @Override
    public Map<String, Object> config() {
        List<Integer> shapeList = new ArrayList<>(shape.length);
        for (int s : shape) shapeList.add(s);
        return Map.of(
                "outputType", outputType.name(),
                "shape", shapeList
                // seed intentionally omitted — data is restored from blob.
        );
    }

    @Override
    public List<NamedTensor> parameters() {
        return List.of(new NamedTensor("data", shape, data.clone()));
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        NamedTensor t = tensors.get("data");
        if (t == null) throw new IllegalArgumentException("Parameter: missing 'data' tensor");
        if (!Arrays.equals(t.shape(), shape)) {
            throw new IllegalArgumentException(
                    "Parameter 'data' shape mismatch: expected " + Arrays.toString(shape)
                            + " got " + Arrays.toString(t.shape()));
        }
        if (t.data().length != data.length) {
            throw new IllegalArgumentException("Parameter 'data' length mismatch");
        }
        System.arraycopy(t.data(), 0, data, 0, data.length);
    }
}
