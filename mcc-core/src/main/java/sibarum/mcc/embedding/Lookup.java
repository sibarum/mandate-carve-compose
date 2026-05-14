package sibarum.mcc.embedding;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code MatrixValue (query) -> StringValue (closest symbol)}.
 *
 * <p>Reverse direction of {@link Embed}. Returns the symbol whose
 * stored embedding has the highest cosine similarity to the query.
 * Read-only against the table; not Trainable.
 */
public final class Lookup implements Primitive {
    private final SymbolEmbeddingTable table;

    public Lookup(SymbolEmbeddingTable table) {
        this.table = table;
    }

    public SymbolEmbeddingTable table() {
        return table;
    }

    @Override
    public String name() {
        return "lookup";
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
                        "lookup: no nearest neighbour (empty table or zero-norm query)"));
        return new StringValue(hit);
    }
}
