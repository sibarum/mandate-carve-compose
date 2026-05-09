package sibarum.strnn.primitive;

import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Picks a specific index from a TokenList. Parameterized by index so different
 * positions show up as distinct transformation-graph nodes.
 */
public final class TokenAt implements Primitive {
    private final int index;

    public TokenAt(int index) {
        this.index = index;
    }

    public int index() {
        return index;
    }

    @Override
    public String name() {
        return "token-at(" + index + ")";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TOKEN_LIST);
    }

    @Override
    public ValueType outputType() {
        return ValueType.STRING;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TokenListValue tl = (TokenListValue) inputs.getFirst();
        if (index >= tl.tokens().size()) {
            return new StringValue("");
        }
        return new StringValue(tl.tokens().get(index));
    }
}
