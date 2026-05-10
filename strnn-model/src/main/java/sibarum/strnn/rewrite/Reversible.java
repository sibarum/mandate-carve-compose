package sibarum.strnn.rewrite;

/**
 * Marker interface for rewrite rules whose
 * {@link RewriteRulePrimitive#inferInput inferInput} is a reliable inverse —
 * applying the rule and then inferring the input recovers the original tree
 * (modulo canonical form). The framework does not enforce this; opt in by
 * implementing the interface, and a runtime round-trip check (see
 * TotalAlgebraDemo) verifies the claim.
 *
 * Opt-in, not required: destructive simplifications (rules that drop
 * information by design) are valid and simply do not claim the marker. A
 * carver, trainer, or downstream pipeline can use {@code instanceof Reversible}
 * to choose between filtering-friendly and reversible rule subsets.
 */
public interface Reversible {
}
