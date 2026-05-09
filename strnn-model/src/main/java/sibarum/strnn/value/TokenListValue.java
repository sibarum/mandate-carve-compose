package sibarum.strnn.value;

import java.util.List;

public record TokenListValue(List<String> tokens) implements Value {
    public TokenListValue {
        tokens = List.copyOf(tokens);
    }

    @Override
    public ValueType type() {
        return ValueType.TOKEN_LIST;
    }
}
