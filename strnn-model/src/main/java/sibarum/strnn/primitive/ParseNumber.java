package sibarum.strnn.primitive;

import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

public final class ParseNumber implements Primitive {
    @Override
    public String name() {
        return "parse-number";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        StringValue s = (StringValue) inputs.getFirst();
        return new NumberValue(Double.parseDouble(s.s()));
    }
}
