package sibarum.mcc.value;

public record NumberValue(double n) implements Value {
    @Override
    public ValueType type() {
        return ValueType.NUMBER;
    }
}
