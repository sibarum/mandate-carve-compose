package sibarum.strnn.cache;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Objects;

/**
 * A {@link Primitive} that wraps a {@link NetworkItem} and exposes it as a
 * single function. From the outer carver's perspective, the cached network
 * is just another deterministic primitive with declared input/output types;
 * the carver can place it in a carving the same way it places any other
 * primitive.
 *
 * This is the seam where Key-Network lives: a stored network is retrieved
 * via the primitive substrate, composed into a larger carving by the
 * carver, and executed as part of the outer pipeline. Multi-cached-network
 * carvings (a chain of two cached networks doing two sequential mappings)
 * are exactly this: two CachedNetworkPrimitive instances wired in series by
 * the carver.
 */
public final class CachedNetworkPrimitive implements Primitive {
    private final String name;
    private final NetworkItem item;

    public CachedNetworkPrimitive(String name, NetworkItem item) {
        this.name = Objects.requireNonNull(name);
        this.item = Objects.requireNonNull(item);
    }

    public NetworkItem item() {
        return item;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(item.inputType());
    }

    @Override
    public ValueType outputType() {
        return item.outputType();
    }

    @Override
    public Value apply(List<Value> inputs) {
        return item.execute(inputs.getFirst());
    }
}
