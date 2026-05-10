# STRNN v2 — Symbolic Rewriting on the Existing Substrate

## What was built

Five phases per `04-v2-plan-symbolic-rewriting.md`, all runnable:

| Phase | Deliverable                                            | Demo                       |
|-------|--------------------------------------------------------|----------------------------|
| 0     | `ParseTreeValue` (sealed: Literal, Variable, BinaryOp), `Operator` enum, `ParseExpression` recursive-descent parser | `ParseTreeDemo`            |
| 1     | `Pattern` (sealed with HolePat for variables), `Bindings`, `Matcher.match` / `Matcher.substitute` | `PatternMatchingDemo`      |
| 2     | `RewriteRulePrimitive` base class + `IdentityZero` (`?a + 0 → ?a`); `TreeOutputPrimitive` and `Terminal` marker | `IdentityZeroDemo`         |
| 3     | `EvaluateBinaryOp(role, mlp)` — learned leaf inside a symbolic rule; implements `Trainable` and `LearnedArithmetic` | `EvaluateBinaryOpDemo`     |
| 4     | Carver inverters extended for tree-shaped targets: `RewriteRulePrimitive`, `EvaluateBinaryOp`, `ParseExpression`, `TreeOutputPrimitive`. `findOutputNode` generalized to dispatch on result-mandate type. `numericAnchors` extended to extract literals from `ParseTreeValue` mandates. Tolerance-aware tree matching in `ValueDistance`. | `TreeCarvingDemo`          |
| 5     | Full rule library (`IdentityOne`, `Distribute`, `FactorCommon` added). Two-variant diagnostic. | `SymbolicRewriteDemo`      |

About 700 lines of new Java across the value, primitive, rewrite, and
carving packages, plus the demos. Below the v2 plan's estimate of
1000-1500 because Distribute and FactorCommon turned out to be cheap
once the pattern-matching substrate was in place.

## The dual-claim diagnostic

The v2 demo's payload is the simplest possible test of the dual claim:
**same library, two inputs, two different carved orchestrations,
distinguished by whether the input requires symbolic or learned
treatment.**

| Variant | Input  | Result mandate                  | Tolerance | Expected primitive | Forbidden primitive |
|---------|--------|---------------------------------|-----------|--------------------|---------------------|
| A       | `x + 0` | `Variable("x")`                | exact     | `identity-zero`    | `evaluate-add`      |
| B       | `3 + 4` | `Literal(7)`                    | 0.5       | `evaluate-add`     | `identity-zero`     |

The structural distinction is not about tolerance or score — it's about
which rule's *pattern* matches. `evaluate-add` requires both children of
the root to be `Literal`; `identity-zero` requires the right child to be
`Literal(0)`. Variant A's parsed tree is `Variable("x") + Literal(0)` —
only `identity-zero` applies. Variant B's parsed tree is
`Literal(3) + Literal(4)` — only `evaluate-add` applies.

**Both variants pass.** Each carving contains its expected primitive
and excludes the forbidden one:

```
Variant A: parse_expr → identity_zero → tree_output    (carved 3 nodes)
Variant B: parse_expr → evaluate-add  → tree_output    (carved 3 nodes)
```

Same primitive library, same carver instance, same seed. The carver
discriminates automatically based on which rules can actually transform
the input toward the result mandate.

## What this licenses

The architecture name earns a literal reading. The framework is, at the
demonstrated level:

- A **symbolic rewrite system**: rewrite rules are first-class
  pattern-defined transformations on tree-shaped values; the carver
  selects rules; mandates can require specific intermediate forms;
  rule application is structurally enforced rather than learned.
- A **heterogeneous network composition platform**: pure-symbolic rules
  (`IdentityZero`, `IdentityOne`, etc.) and learned-leaf rules
  (`EvaluateBinaryOp` with an MLP inside) coexist in the same library
  and are dispatched by the same carver under the same action principle.
  `EvaluateBinaryOp` is the load-bearing primitive — a learned
  component lives literally *inside* a rewrite rule, with the rule's
  pattern controlling when the learned component runs.

These are not two systems sharing infrastructure; they are one system
where the primitive library happens to mix rule-based and learned
components.

The v0 + Pattern A + Pattern B work was about the substrate. v2 added
the content that makes the substrate actually *symbolic*.

## What v2 does NOT demonstrate

- **Sub-tree rule application.** All rules in v2 apply at the root of
  the input tree. `Distribute` and `FactorCommon` are in the library
  but are not exercised by the two-variant demo, because exercising
  them would require evaluating the sub-trees of a distributed
  expression — and our learned-leaf rule only applies at root.
  Composition like `parse → Distribute → eval-mul → eval-mul →
  eval-add → output` for `(2+3)*4` would require either:
    1. A descent strategy that lets `eval-mul` walk into a sub-tree, or
    2. A sub-tree extraction primitive (`LeftChild`, `RightChild`) so
       the carver can manually thread sub-trees through evaluation.
  Both are v3 questions.

- **Rule-search at meaningful scale.** The two variants are tiny
  3-node carvings. Whether the action principle's mandate-driven
  search remains tractable with a larger rule library — say, 20-50
  rules including conditional and multi-pattern rules — is unanswered.
  The §10.1 stability concern from the original design doc applies here
  more than it did in v0/v1, because rule-application search spaces
  grow combinatorially.

- **Action-principle-driven rule preference.** The carving was a single
  shot per variant; no edge-stats accumulation, no pruning, no
  Pattern-B-style competitive selection between alternative rules. With
  many rules competing at each tree position, the ε-greedy + edge-stats
  machinery from v1 is exactly what would govern selection — but v2
  doesn't run it.

- **Symbolic invariants beyond pattern matching.** Real symbolic rewrite
  systems handle equational reasoning, congruence, normalization,
  decision procedures. v2 stays in the unidirectional-rewrite world.
  The framework could extend toward those (mandates can express more
  than just "produce this term"), but v2 hasn't tried.

## What the diagnostic actually proves

It is honest to phrase v2's result narrowly:

> **Given a primitive library that mixes symbolic rewrite rules and
> learned-leaf evaluators with the same type signature, the carver
> selects the structurally appropriate primitive based on which rule's
> pattern matches the parsed input. Mandates can require specific
> intermediate or final tree shapes, and rule application is
> structurally enforced.**

That is a real demonstration of the dual claim — *for one operator,
one rewrite rule per kind, and root-level rule application*. It is
not a demonstration that the framework would handle full algebraic
simplification, automated theorem proving, or any non-trivial
symbolic-rewriting task. Those would need the v3 sub-tree extension
plus a richer rule library.

The architecture's name now describes what exists, not just what was
intended. The frame "*symbolic rewrite system that doubles as a
heterogeneous network composition platform*" earns its quotation
marks. Whether it scales to interesting symbolic tasks is what v3
would test.

## Concrete v3 questions

Three questions, in dependency order, that v3 would address:

1. **Sub-tree rule application.** How should rules be invoked on
   sub-trees? Two options:
   - A `Recurse` strategy primitive that descends into a child and
     applies a chosen rule there.
   - Built-in rule-walking inside each rewrite primitive (innermost
     leftmost or outermost). This couples the strategy to the rule.
   The first is more orthogonal but adds primitives; the second is
   simpler but less flexible. v3 should pick one and stick with it.

2. **Rule-application search at scale.** With 20+ rules and sub-tree
   application, the carver's backward search may hit budget caps. Worth
   instrumenting per-step search cost early. May also need
   `(target-shape, primitiveClass)` cycle detection rather than just
   `(target-tree-instance, primitiveClass)` — structural hashing may
   help.

3. **Where do mandates come from for rich symbolic tasks?** v2's
   mandates were hand-specified. For a useful tool, the engineer
   shouldn't need to specify every intermediate. Whether mandates can
   be *derived* from a high-level specification (e.g., "produce a
   normal form" rather than "produce this specific tree") is a
   question about the framework's user-facing surface. The §6.7
   curriculum-relaxation idea applies: start with strong mandates,
   relax over time.

## Bottom line

v2 went well in the same sense v0 and Pattern A/B went well: we now
know more sharply what to ask next. The dual claim earns its name on
the simplest test, and the path to harder tests (sub-tree application,
rule-library scale, derived mandates) is clear. The substrate built in
v0 and refined in v1 supported the v2 work without architectural
change — which is itself meaningful evidence that the original design's
abstractions were the right ones.

The project's name has caught up to its implementation.
