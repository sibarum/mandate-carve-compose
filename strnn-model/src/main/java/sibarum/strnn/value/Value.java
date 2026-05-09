package sibarum.strnn.value;

public sealed interface Value
        permits StringValue, TokenListValue, NumberValue, MatrixValue {
    ValueType type();
}
