package sibarum.mcc.op;

import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;
import java.util.Map;

/**
 * Primitive: take a contiguous slice {@code [from, to)} of an input
 * vector. Configured at construction (the slice range is a structural
 * parameter of the node, not a runtime input).
 *
 * <p>Backward: zero-pad. {@code dL/dx[from:to] = gradOut};
 * {@code dL/dx[other indices] = 0}.
 */
public final class Slice implements Differentiable, Configurable {

    private final int from;
    private final int to;
    private int lastInputLength;

    public Slice(int from, int to) {
        if (from < 0 || to < from) {
            throw new IllegalArgumentException(
                    "Slice requires 0 <= from <= to, got from=" + from + ", to=" + to);
        }
        this.from = from;
        this.to = to;
    }

    public int from() { return from; }
    public int to() { return to; }

    @Override
    public String name() {
        return "slice";
    }

    @Override
    public Map<String, Object> config() {
        return Map.of("from", from, "to", to);
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
        if (to > x.length) {
            throw new IllegalArgumentException(
                    "Slice[" + from + ":" + to + "] exceeds input length " + x.length);
        }
        double[] out = new double[to - from];
        System.arraycopy(x, from, out, 0, to - from);
        lastInputLength = x.length;
        return new MatrixValue(out);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastInputLength == 0) throw new IllegalStateException("Slice backward without prior apply");
        double[] g = ((MatrixValue) gradOutput).data();
        if (g.length != to - from) {
            throw new IllegalArgumentException("Slice gradOutput dim mismatch");
        }
        double[] dx = new double[lastInputLength];
        System.arraycopy(g, 0, dx, from, g.length);
        return List.of(new MatrixValue(dx));
    }
}
