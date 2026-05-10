package sibarum.strnn.rewrite;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.LearnedArithmetic;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.Trainable;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * The heterogeneity primitive: a symbolic rewrite rule with a learned-leaf
 * computation. Pattern-matches on {@code Lit(?a) op Lit(?b)}; produces
 * {@code Lit(?a op ?b)} where the actual numeric value comes from a wrapped
 * Mlp running on the (a, b) pair in the same NumberToMatrix-scaled encoding
 * the v1 demos used.
 *
 * Implements LearnedArithmetic so the carver's existing inverter and Pattern
 * B's primitive-competition pruning work without modification — the role tag
 * is identical to MlpPrimitive's, the type signature differs (PARSE_TREE
 * here, MATRIX there) but the carving machinery dispatches by primitive class
 * type, not by signature.
 */
public final class EvaluateBinaryOp implements Trainable, LearnedArithmetic {
    private static final double SCALE = NumberToMatrix.SCALE;

    private final MlpRole role;
    private final Mlp mlp;

    public EvaluateBinaryOp(MlpRole role, Mlp mlp) {
        this.role = role;
        this.mlp = mlp;
    }

    @Override
    public MlpRole role() {
        return role;
    }

    public Mlp mlp() {
        return mlp;
    }

    @Override
    public String name() {
        return "evaluate-" + role.name().toLowerCase();
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.PARSE_TREE);
    }

    @Override
    public ValueType outputType() {
        return ValueType.PARSE_TREE;
    }

    @Override
    public Value apply(List<Value> inputs) {
        ParseTreeValue tree = (ParseTreeValue) inputs.getFirst();
        if (!(tree instanceof ParseTreeValue.BinaryOp bo)) {
            throw new IllegalArgumentException(
                    name() + " expected a BinaryOp at root, got " + tree);
        }
        if (bo.op() != asOperator()) {
            throw new IllegalArgumentException(
                    name() + " expected operator " + asOperator() + ", got " + bo.op());
        }
        if (!(bo.left() instanceof ParseTreeValue.Literal la)
                || !(bo.right() instanceof ParseTreeValue.Literal lb)) {
            throw new IllegalArgumentException(
                    name() + " expected literal children, got " + tree);
        }
        double[] in = new double[]{la.value() / SCALE, lb.value() / SCALE};
        double result = mlp.forward(in)[0] * SCALE;
        return new ParseTreeValue.Literal(result);
    }

    @Override
    public void backward(Value target) {
        if (!(target instanceof ParseTreeValue.Literal lit)) {
            throw new IllegalArgumentException(
                    name() + " backward expects a Literal target, got " + target);
        }
        double[] mlpTarget = new double[]{lit.value() / SCALE};
        mlp.backward(mlpTarget);
    }

    @Override
    public void step(double lr) {
        mlp.step(lr);
    }

    @Override
    public Object trainableIdentity() {
        return mlp;
    }

    private sibarum.strnn.value.Operator asOperator() {
        return switch (role) {
            case ADD -> sibarum.strnn.value.Operator.ADD;
            case MUL -> sibarum.strnn.value.Operator.MUL;
        };
    }
}
