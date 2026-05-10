package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code (NumberValue, NumberValue) -> NumberValue} subtraction.
 *
 * Routed through {@link TotalArithmetic#totalSub} so {@code ±∞ − ±∞ = 0}
 * (same sign) — total at the scalar level just like the vector ops.
 */
public final class NumberSub implements Primitive {

    @Override
    public String name() {
        return "number-sub";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.NUMBER, ValueType.NUMBER);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        double a = ((NumberValue) inputs.get(0)).n();
        double b = ((NumberValue) inputs.get(1)).n();
        return new NumberValue(TotalArithmetic.totalSub(a, b));
    }
}
