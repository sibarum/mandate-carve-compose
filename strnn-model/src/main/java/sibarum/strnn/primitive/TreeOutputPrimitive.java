package sibarum.strnn.primitive;

import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Identity passthrough that marks the terminal node of a ParseTree-typed
 * computation graph. Parallels OutputPrimitive for v0's NumberValue chain;
 * v2 needs both since the result mandate's type drives terminal selection.
 */
public final class TreeOutputPrimitive implements Terminal {
    @Override
    public String name() {
        return "tree-output";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.PARSE_TREE);
    }

    @Override
    public ValueType outputType() {
        return ValueType.PARSE_TREE;
    }

    @Override
    public Value apply(List<Value> inputs) {
        return inputs.getFirst();
    }
}
