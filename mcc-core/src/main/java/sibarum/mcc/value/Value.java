package sibarum.mcc.value;

/**
 * Sealed root of the mcc-core value hierarchy. Every primitive's input
 * and output is a {@code Value}. New value types must be added to the
 * permits list and to {@link ValueType}.
 */
public sealed interface Value
        permits StringValue, NumberValue, MatrixValue,
                TernionValue, QuaternionValue, TensorValue {
    ValueType type();
}
