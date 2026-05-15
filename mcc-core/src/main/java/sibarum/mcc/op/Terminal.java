package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.primitive.Inversion;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Identity passthrough used as the terminal node of a carved graph.
 * One {@code Terminal} exists per output {@link ValueType} of interest
 * — the carver locates it by matching the result mandate's type.
 *
 * <p>Inversion is exact identity: given a desired output value, the
 * required input is the same value.
 */
public final class Terminal implements Differentiable {

    private final ValueType type;

    public Terminal(ValueType type) {
        this.type = type;
    }

    public ValueType type() {
        return type;
    }

    @Override
    public String name() {
        return "terminal-" + type.name().toLowerCase();
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(type);
    }

    @Override
    public ValueType outputType() {
        return type;
    }

    @Override
    public Value apply(List<Value> inputs) {
        return inputs.getFirst();
    }

    @Override
    public Inversion inversion() {
        return (target, ctx) -> List.of(target);
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        // Identity passthrough — gradient flows unchanged.
        return List.of(gradOutput);
    }
}
