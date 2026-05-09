package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

public interface Primitive {
    String name();

    List<ValueType> inputTypes();

    ValueType outputType();

    Value apply(List<Value> inputs);
}
