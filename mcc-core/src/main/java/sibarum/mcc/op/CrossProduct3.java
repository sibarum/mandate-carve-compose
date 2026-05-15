package sibarum.mcc.op;

import sibarum.mcc.primitive.Differentiable;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.List;

/**
 * Primitive: 3D cross product on {@link TernionValue}.
 * {@code (u, v) -> (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁)}.
 *
 * <p>Components are interpreted as a 3D Euclidean vector.
 *
 * <p>Backward (for bilinear cross product {@code y = u × v}):
 * {@code dL/du = v × gradOut}, {@code dL/dv = gradOut × u}.
 */
public final class CrossProduct3 implements Differentiable {

    private TernionValue lastU;
    private TernionValue lastV;

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
        lastU = a;
        lastV = b;
        return new TernionValue(
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x()
        );
    }

    @Override
    public List<Value> backward(Value gradOutput) {
        if (lastU == null) throw new IllegalStateException("CrossProduct3 backward without prior apply");
        TernionValue g = (TernionValue) gradOutput;
        TernionValue dU = cross(lastV, g);
        TernionValue dV = cross(g, lastU);
        return List.of(dU, dV);
    }

    private static TernionValue cross(TernionValue a, TernionValue b) {
        return new TernionValue(
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x()
        );
    }
}
