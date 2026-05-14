package sibarum.mcc.op;

import sibarum.mcc.primitive.Primitive;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: 3D cross product on {@link TernionValue}.
 * {@code (u, v) -> (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁)}.
 *
 * <p>The components are interpreted as a 3D Euclidean vector. For
 * non-Euclidean uses of {@code TernionValue} (e.g. three independent
 * scalars), pick a different primitive.
 */
public final class CrossProduct3 implements Primitive {

    @Override
    public String name() {
        return "cross-product-3";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.TERNION, ValueType.TERNION);
    }

    @Override
    public ValueType outputType() {
        return ValueType.TERNION;
    }

    @Override
    public Value apply(List<Value> inputs) {
        TernionValue a = (TernionValue) inputs.get(0);
        TernionValue b = (TernionValue) inputs.get(1);
        return new TernionValue(
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x()
        );
    }
}
