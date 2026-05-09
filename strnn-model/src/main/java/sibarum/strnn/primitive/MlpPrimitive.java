package sibarum.strnn.primitive;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Has two modes:
 *  - <i>Stub mode</i> (constructor with no Mlp): hardcodes the arithmetic the
 *    role would eventually learn, so the surrounding pipeline can be exercised
 *    without training. Used in Phase 0.
 *  - <i>Trainable mode</i> (constructor with Mlp): forwards through the supplied
 *    network. backward() / step() apply gradients. Used in Phase 1+.
 *
 * Two MlpPrimitive instances with the same role share an Mlp by passing the
 * same Mlp reference into both constructors — that is the v0 sharing
 * mechanism the plan calls out as a guess flag.
 */
public final class MlpPrimitive implements Trainable {
    private final MlpRole role;
    private final Mlp mlp;

    /** Stub mode for Phase 0. */
    public MlpPrimitive(MlpRole role) {
        this.role = role;
        this.mlp = null;
    }

    /** Trainable mode for Phase 1+; pass the same Mlp to all instances of this role to share weights. */
    public MlpPrimitive(MlpRole role, Mlp mlp) {
        this.role = role;
        this.mlp = mlp;
    }

    public MlpRole role() {
        return role;
    }

    public boolean isStub() {
        return mlp == null;
    }

    public Mlp mlp() {
        return mlp;
    }

    @Override
    public String name() {
        return "mlp(" + role.name().toLowerCase() + ")";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.MATRIX);
    }

    @Override
    public ValueType outputType() {
        return ValueType.MATRIX;
    }

    @Override
    public Value apply(List<Value> inputs) {
        MatrixValue in = (MatrixValue) inputs.getFirst();
        if (in.dim() != 2) {
            throw new IllegalArgumentException(
                    "MlpPrimitive expects a 2-dim composed matrix, got dim=" + in.dim());
        }
        if (mlp == null) {
            double a = in.data()[0];
            double b = in.data()[1];
            double scale = NumberToMatrix.SCALE;
            double out = switch (role) {
                case ADD -> a + b;
                case MUL -> a * b * scale;
            };
            return new MatrixValue(new double[]{out});
        }
        return new MatrixValue(mlp.forward(in.data()));
    }

    @Override
    public void backward(Value target) {
        if (mlp == null) return;
        MatrixValue t = (MatrixValue) target;
        mlp.backward(t.data());
    }

    @Override
    public void step(double lr) {
        if (mlp == null) return;
        mlp.step(lr);
    }
}
