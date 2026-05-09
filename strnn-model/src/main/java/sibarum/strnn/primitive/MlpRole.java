package sibarum.strnn.primitive;

/**
 * Semantic role tag for MLP primitive sharing. Two MlpPrimitive instances with
 * the same role share weights; different roles get separate weight pools. The
 * tag is the v0 mechanism for &quot;which MLP is this&quot; and corresponds to the
 * sharing-granularity guess flag in the plan.
 */
public enum MlpRole {
    ADD,
    MUL
}
