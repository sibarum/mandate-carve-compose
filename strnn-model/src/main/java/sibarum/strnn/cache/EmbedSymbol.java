package sibarum.strnn.cache;

import sibarum.strnn.primitive.Trainable;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Primitive: {@code StringValue (symbol) -> MatrixValue (embedding)}.
 *
 * Reads from a shared {@link SymbolEmbeddingTable}. Multiple EmbedSymbol
 * primitives sharing the same table return the same {@link #trainableIdentity}
 * — the trainer's IdentityHashMap-keyed dedup ensures each unique table is
 * stepped once per training example regardless of how many EmbedSymbol
 * instances call apply.
 *
 * Backward semantics: given a target vector v*, gradient is (output - v*),
 * applied by {@code step(lr)} as a simple SGD update directly on the
 * embedding row for the most recent symbol.
 */
public final class EmbedSymbol implements Trainable {
    private final SymbolEmbeddingTable table;
    private String lastSymbol;
    private double[] lastOutput;
    private double[] pendingGradient;

    public EmbedSymbol(SymbolEmbeddingTable table) {
        this.table = table;
    }

    @Override
    public String name() {
        return "embed-symbol";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.STRING);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        StringValue s = (StringValue) inputs.getFirst();
        lastSymbol = s.s();
        double[] v = table.embed(lastSymbol);
        lastOutput = v.clone();
        return new MatrixValue(v);
    }

    @Override
    public void backward(Value target) {
        if (lastSymbol == null) {
            throw new IllegalStateException("backward called without prior apply");
        }
        if (!(target instanceof MatrixValue m)) {
            throw new IllegalArgumentException("EmbedSymbol target must be MatrixValue");
        }
        double[] t = m.data();
        if (t.length != lastOutput.length) {
            throw new IllegalArgumentException(
                    "target dim " + t.length + " != output dim " + lastOutput.length);
        }
        pendingGradient = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            pendingGradient[i] = lastOutput[i] - t[i];
        }
    }

    @Override
    public void step(double lr) {
        if (lastSymbol == null || pendingGradient == null) return;
        table.update(lastSymbol, pendingGradient, lr);
        pendingGradient = null;
    }

    @Override
    public Object trainableIdentity() {
        return table;
    }
}
