package sibarum.strnn.primitive;

import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.List;

public final class SplitStringAt implements Primitive {
    private final char delim;

    public SplitStringAt(char delim) {
        this.delim = delim;
    }

    public char delim() {
        return delim;
    }

    @Override
    public String name() {
        return "split-string-at('" + delim + "')";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TOKEN_LIST;
    }

    @Override
    public Value apply(List<Value> inputs) {
        StringValue s = (StringValue) inputs.getFirst();
        List<String> tokens = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < s.s().length(); i++) {
            char c = s.s().charAt(i);
            if (c == delim) {
                tokens.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        tokens.add(cur.toString());
        return new TokenListValue(tokens);
    }
}
