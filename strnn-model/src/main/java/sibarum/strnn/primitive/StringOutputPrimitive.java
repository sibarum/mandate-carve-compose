package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Terminal that passes a {@link sibarum.strnn.value.StringValue} through.
 * Mirrors {@link OutputPrimitive} (NumberValue) and TreeOutputPrimitive
 * (ParseTreeValue) — the String analogue, needed for carvings whose
 * terminal is a symbol.
 */
public final class StringOutputPrimitive implements Terminal {

    @Override
    public String name() {
        return "string-output";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.STRING;
    }

    @Override
    public Value apply(List<Value> inputs) {
        return inputs.getFirst();
    }
}
