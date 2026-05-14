package sibarum.mcc.value;

public record StringValue(String s) implements Value {
    @Override
    public ValueType type() {
        return ValueType.STRING;
    }
}
