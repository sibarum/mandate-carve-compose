package sibarum.mcc.embedding;

import sibarum.mcc.primitive.Configurable;
import sibarum.mcc.primitive.Inversion;
import sibarum.mcc.primitive.Parameterized;
import sibarum.mcc.primitive.Trainable;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Primitive: {@code StringValue (symbol) -> MatrixValue (embedding)}.
 *
 * <p>Reads from a shared {@link SymbolEmbeddingTable}. Multiple
 * {@code Embed} primitives sharing the same table return the same
 * {@link #trainableIdentity} — the trainer's IdentityHashMap-keyed
 * dedup ensures each unique table is stepped once per training
 * example regardless of how many {@code Embed} instances call apply.
 *
 * <p>Backward semantics: given a target vector v*, gradient is
 * {@code (output − v*)}, applied by {@code step(lr)} as a direct SGD
 * update on the embedding row for the most recent symbol.
 */
public final class Embed implements Trainable, Parameterized, Configurable {
    private final SymbolEmbeddingTable table;
    private String lastSymbol;
    private double[] lastOutput;
    private double[] pendingGradient;

    public Embed(SymbolEmbeddingTable table) {
        this.table = table;
    }

    public SymbolEmbeddingTable table() {
        return table;
    }

    @Override
    public String name() {
        return "embed";
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
    public List<Value> backward(Value gradOutput) {
        if (lastSymbol == null) {
            throw new IllegalStateException("backward called without prior apply");
        }
        if (!(gradOutput instanceof MatrixValue mg)) {
            throw new IllegalArgumentException("Embed gradOutput must be MatrixValue");
        }
        double[] g = mg.data();
        if (g.length != lastOutput.length) {
            throw new IllegalArgumentException(
                    "gradOutput dim " + g.length + " != output dim " + lastOutput.length);
        }
        // Accumulate so multiple apply→backward calls between steps sum.
        if (pendingGradient == null) pendingGradient = new double[g.length];
        for (int i = 0; i < g.length; i++) pendingGradient[i] += g[i];
        // The input is a StringValue — no continuous gradient flows back.
        return java.util.Collections.singletonList(null);
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

    @Override
    public Inversion inversion() {
        // Embed maps a known symbol → its stored vector. Given a target
        // vector, the closest stored symbol (by cosine) is the natural
        // inversion. If the table is empty or the query has zero norm
        // the carver should treat this as a dead end.
        return (target, ctx) -> {
            if (!(target instanceof MatrixValue mv)) return null;
            return table.nearest(mv.data())
                    .map(s -> List.<Value>of(new StringValue(s)))
                    .orElse(null);
        };
    }

    @Override
    public Map<String, Object> config() {
        // Symbols are part of the structural state of the table, so they
        // belong in config (not in the params blob). The vectors themselves
        // ride in parameters() as one [N, dim] tensor.
        return Map.of(
                "dim", table.dim(),
                "symbols", new ArrayList<>(table.symbols())
                // seed intentionally omitted — vectors are restored from blob.
        );
    }

    @Override
    public List<NamedTensor> parameters() {
        List<String> symbols = new ArrayList<>(table.symbols());
        int n = symbols.size();
        int dim = table.dim();
        double[] flat = new double[n * dim];
        for (int i = 0; i < n; i++) {
            double[] v = table.rawVector(symbols.get(i));
            System.arraycopy(v, 0, flat, i * dim, dim);
        }
        return List.of(new NamedTensor("embeddings", new int[] { n, dim }, flat));
    }

    @Override
    public void loadParameters(Map<String, NamedTensor> tensors) {
        NamedTensor t = tensors.get("embeddings");
        if (t == null) throw new IllegalArgumentException("Embed: missing 'embeddings' tensor");
        List<String> symbols = new ArrayList<>(table.symbols());
        int n = symbols.size();
        int dim = table.dim();
        if (!Arrays.equals(t.shape(), new int[] { n, dim })) {
            throw new IllegalArgumentException(
                    "Embed 'embeddings' shape mismatch: expected "
                            + Arrays.toString(new int[] { n, dim })
                            + " got " + Arrays.toString(t.shape())
                            + " (the registry must pre-populate the table with the same symbols)");
        }
        for (int i = 0; i < n; i++) {
            double[] row = new double[dim];
            System.arraycopy(t.data(), i * dim, row, 0, dim);
            table.put(symbols.get(i), row);
        }
    }
}
