package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code MatrixValue (query) -> StringValue (closest symbol)}.
 *
 * The reverse direction of {@link EmbedSymbol}. Returns the symbol whose
 * stored embedding has the highest cosine similarity to the query. Read-only
 * against the table; not Trainable.
 */
public final class LookupSymbol implements Primitive {
    private final SymbolEmbeddingTable table;

    public LookupSymbol(SymbolEmbeddingTable table) {
        this.table = table;
    }

    @Override
    public String name() {
        return "lookup-symbol";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.STRING;
    }

    @Override
    public Value apply(List<Value> inputs) {
        MatrixValue m = (MatrixValue) inputs.getFirst();
        String hit = table.nearest(m.data())
                .orElseThrow(() -> new IllegalStateException(
                        "lookup-symbol: no nearest neighbour (empty table or zero-norm query)"));
        return new StringValue(hit);
    }
}
