package sibarum.strnn.rewrite;

import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Optional;

/**
 * Base class for pure-symbolic rewrite primitives. A rule has a name, a
 * left-hand-side pattern (matched against the input), and a right-hand-side
 * pattern (used to produce the rewritten output by substituting the bindings
 * captured during matching).
 *
 * apply(input) requires the input to match the LHS; throws otherwise. The
 * carver's inverter is expected to ensure applicability before placement.
 *
 * Subclasses for each rule (IdentityZero, IdentityOne, Distribute, etc.).
 */
public abstract class RewriteRulePrimitive implements Primitive {
    private final String name;
    private final Pattern lhs;
    private final Pattern rhs;

    protected RewriteRulePrimitive(String name, Pattern lhs, Pattern rhs) {
        this.name = name;
        this.lhs = lhs;
        this.rhs = rhs;
    }

    public Pattern lhs() {
        return lhs;
    }

    public Pattern rhs() {
        return rhs;
    }

    @Override
    public final String name() {
        return name;
    }

    @Override
    public final List<ValueType> inputTypes() {
        return List.of(ValueType.PARSE_TREE);
    }

    @Override
    public final ValueType outputType() {
        return ValueType.PARSE_TREE;
    }

    @Override
    public final Value apply(List<Value> inputs) {
        ParseTreeValue tree = (ParseTreeValue) inputs.getFirst();
        Optional<Bindings> bindings = Matcher.match(lhs, tree);
        if (bindings.isEmpty()) {
            throw new IllegalArgumentException(
                    name + " did not match input tree: " + tree);
        }
        return Matcher.substitute(rhs, bindings.get());
    }

    /** Whether this rule can apply to the given tree. */
    public final boolean applies(ParseTreeValue tree) {
        return Matcher.match(lhs, tree).isPresent();
    }

    /**
     * Inverter helper for the carver: given a desired output tree, attempt to
     * compute an input tree that this rule would rewrite to that output.
     * Inversion succeeds when the output matches the RHS pattern; bindings
     * captured there are substituted into the LHS to produce the input.
     *
     * Returns empty if the output does not match the RHS pattern.
     *
     * For non-unique inversions (rules where the LHS introduces holes not
     * present in the RHS, e.g. {@code x → x + 0}), the LHS substitution would
     * fail with a missing-binding error. Subclasses can override
     * {@link #inferInputs(ParseTreeValue)} to enumerate possibilities.
     */
    public Optional<ParseTreeValue> inferInput(ParseTreeValue output) {
        Optional<Bindings> b = Matcher.match(rhs, output);
        if (b.isEmpty()) return Optional.empty();
        try {
            return Optional.of(Matcher.substitute(lhs, b.get()));
        } catch (IllegalStateException e) {
            return Optional.empty();
        }
    }
}
