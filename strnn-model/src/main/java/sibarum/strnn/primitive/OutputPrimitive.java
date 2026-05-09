package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Identity passthrough that marks the terminal node of a computation graph.
 */
public final class OutputPrimitive implements Primitive {
    @Override
    public String name() {
        return "output";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.NUMBER);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        return inputs.getFirst();
    }
}
