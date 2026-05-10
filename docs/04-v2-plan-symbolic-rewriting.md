# STRNN v2 Plan — Symbolic Term Rewriting on the Existing Substrate

## Context

The v0 + v1 work demonstrated:

- **Mandate-enforced interpretability** (v0): the carver can be required
  to materialize specific intermediate values; failure to do so is
  diagnosable.
- **Component-agnostic orchestration** (Pattern A): swapping the
  underlying learner (MLP ↔ transformer) at the same role and signature
  leaves the carved orchestration structurally identical.
- **Quality-responsive selection** (Pattern B + ε-greedy): when
  competing primitives fill the same role, edge stats reflect quality
  and pruning correctly identifies underperformers — *provided
  exploration is enabled*.

What's *not* demonstrated, and what gives the project's name its
aspirational character: that the architecture is a **symbolic term
rewrite system** — that it can apply pattern-matched rewrite rules to
structured terms, derive new symbolic forms via rule application, and
do so under the same orchestration and action-principle machinery that
already handles learned components.

v2's purpose is to crystallize the dual claim: **the architecture is at
once a symbolic term rewrite system *and* a heterogeneous network
composition platform, sharing one orchestration substrate.**

## The dual claim, made precise

The two halves are not separate systems sharing infrastructure. They
are the same system applied to two kinds of primitives in the same
library at once:

- **Symbolic rewrite primitives** match patterns on tree-shaped terms
  and produce rewritten terms. Type signature: `ParseTree → ParseTree`.
  Deterministic, but with non-unique inversion (multiple inputs can
  rewrite to the same output).
- **Learned-leaf primitives** evaluate sub-expressions to literal
  values. Type signature: `ParseTree → ParseTree` (where the output is
  a literal-leaf tree). Internal structure is an MLP or transformer.

The same carver, the same action principle, the same mandate
verification, the same edge stats. The only thing that changes is what
sits inside the primitive boxes and what the carver's inverters need
to handle.

## Five deliverables, in dependency order

### 1. `ParseTreeValue` — the missing structured-value type

A new `Value` permitted by the sealed `Value` interface:

```java
public sealed interface ParseTreeValue extends Value {
    record Literal(double value) implements ParseTreeValue {}
    record Variable(String name) implements ParseTreeValue {}
    record BinaryOp(Operator op, ParseTreeValue left, ParseTreeValue right) implements ParseTreeValue {}
    enum Operator { ADD, MUL, SUB, DIV }
}
```

Equality is structural; `ValueDistance` for trees is exact-match (1.0
for unequal, 0.0 for equal). v0's "values as graphs" property (§2.4)
becomes operational rather than aspirational.

Estimated size: ~80 lines.

### 2. `Pattern` and pattern-matching substrate

A `Pattern` is itself a parse tree with *holes* (typed variables that
bind during match):

```java
public sealed interface Pattern {
    record LitPat(double value) implements Pattern {}
    record VarPat(String name) implements Pattern {}
    record HolePat(String name, Class<? extends ParseTreeValue> type) implements Pattern {}
    record OpPat(Operator op, Pattern left, Pattern right) implements Pattern {}
}

public final class Matcher {
    public static Optional<Bindings> match(Pattern p, ParseTreeValue t);
    public static ParseTreeValue substitute(Pattern p, Bindings b);
}
```

`Bindings` is a map from hole name to bound subtree. Standard symbolic
AI machinery; ~150 lines.

This is what makes a rewrite primitive a *rule* and not a hand-coded
function. Without patterns, "symbolic" is just "structured values";
with patterns, the rules are first-class objects the framework can
reason about.

### 3. A small library of rewrite primitives

Five rules is enough to be diagnostic:

| Primitive             | Pattern                     | Rewrite                | Notes                                      |
|-----------------------|-----------------------------|------------------------|--------------------------------------------|
| `Distribute`          | `(?a + ?b) * ?c`            | `?a*?c + ?b*?c`        | Pure symbolic; doubles size                |
| `FactorCommon`        | `?a*?c + ?b*?c`             | `(?a + ?b) * ?c`       | Pure symbolic; inverse of Distribute       |
| `IdentityZero`        | `?a + 0`                    | `?a`                   | Pure symbolic; canonical-form rule         |
| `IdentityOne`         | `?a * 1`                    | `?a`                   | Pure symbolic                              |
| `EvaluateBinaryOp`    | `Lit(?a) op Lit(?b)`        | `Lit(?a op ?b)`        | **Heterogeneous: leaf op via MLP**         |

Each rule is a `Primitive` with type signature
`(ParseTreeValue) → ParseTreeValue`. It applies its pattern at the
*root* of the input tree (the carver decides which sub-tree to apply
it to by structuring the chain). For v2 we keep it root-only; sub-tree
application can be a v3 question.

`EvaluateBinaryOp` is the load-bearing one for the dual claim. It
matches when both children are literal leaves and uses an MLP/
transformer to compute the result. That is a *learned component
embedded inside a symbolic rewrite rule*, selected by the same carver
that selects pure-symbolic rules. Pattern A's `LearnedArithmetic`
machinery extends here naturally.

Each primitive: ~30-50 lines.

### 4. Carver inverters for tree-shaped values

Given a target `ParseTreeValue` T and a candidate rewrite primitive P,
what input tree(s) would P rewrite to T?

| Rewrite              | Inverter logic                                                                                  |
|----------------------|-------------------------------------------------------------------------------------------------|
| `Distribute`         | If T matches `?a*?c + ?b*?c` (i.e., FactorCommon's pattern), bindings give the input            |
| `FactorCommon`       | If T matches `(?a + ?b) * ?c`, bindings give the input                                          |
| `IdentityZero`       | T → `T + 0` (non-unique: also `0 + T`)                                                          |
| `IdentityOne`        | T → `T * 1` (non-unique: also `1 * T`)                                                          |
| `EvaluateBinaryOp`   | Lit(v) ← `Lit(?a) op Lit(?b)` for any pair (a, b) with a op b ≈ v from the numeric anchor pool  |

The pattern-matching substrate makes most of these mechanical: each
rule's inverter is just running the *output* pattern as a matcher and
binding the same holes. `EvaluateBinaryOp` reuses the existing
`LearnedArithmetic` numeric-anchor logic, so inverter coupling between
the carver and the learned-component world stays as it is now.

The non-unique cases (`IdentityZero`, `IdentityOne`) are handled the
same way `MlpPrimitive`'s ADD inverter handles non-unique solutions:
enumerate options, let the carver try them in order. Cycle prevention
becomes more important here: `?a → ?a + 0 → ?a + 0 + 0` is a real loop
risk.

Estimated size: ~100-200 lines total across all rules.

### 5. A demo task that exercises both halves

The right shape: an expression where some sub-evaluations need symbolic
manipulation to become tractable, and the leaf arithmetic is learned.

**Concrete demo:**

```
Input:    "(2 + 3) * 4"  parsed into ParseTreeValue
Target:   Lit(20)
Mandates: - intermediate: a tree of the form Lit(?) * Lit(?) must appear
                          (forces Distribute or full evaluation of the sum)
          - intermediate: at least one Lit(?) tree containing the value 5
                          (forces evaluation of the inner add)
          - result: Lit(20)
```

The carver has at least two valid orchestrations:

- **Direct evaluation**: parse → `EvaluateBinaryOp`(2+3=5) →
  `EvaluateBinaryOp`(5*4=20). Two MLP calls, no symbolic rewriting.
  Materializes the `Lit(5)` mandate naturally.
- **Distribute-then-evaluate**: parse → `Distribute`((2+3)*4 →
  `2*4 + 3*4`) → `EvaluateBinaryOp`(2*4=8) → `EvaluateBinaryOp`(3*4=12)
  → `EvaluateBinaryOp`(8+12=20). Three MLP calls, one symbolic step.
  Materializes the `Lit(_)*Lit(_)` mandate naturally.

Both satisfy all mandates. The carver picks based on edge stats.

Two further variants make the test sharper:

- **Variant A (mandate forces symbolic rewriting):** require the
  intermediate `Lit(2)*Lit(4) + Lit(3)*Lit(4)` to appear. Now only
  `Distribute` paths satisfy. If the carver finds them, symbolic
  rewriting is doing real work.
- **Variant B (mandate forces direct evaluation):** require the
  intermediate `Lit(5) * Lit(4)` to appear. Now only direct paths
  satisfy. Tests whether the carver can decline symbolic rewriting
  when it isn't useful.

Variants A and B together exercise the framework's discrimination:
the same primitive library, different mandates, different carved
structures. That's the dual claim demonstrated empirically.

## What this would crystallize

If the deliverables work and Variants A and B both pass:

- *Symbolic rewrite system*: rules express tree-to-tree transformations
  via patterns; the carver selects rules; mandates can require specific
  intermediate forms; new symbolic terms are derived by rule
  application.
- *Heterogeneous composition platform*: `EvaluateBinaryOp` carries a
  learned component (MLP/transformer) inside a symbolic rewrite rule;
  pure-symbolic and learned-leaf rules are selected by the same carver
  under the same action principle; mandates define typed interfaces
  between them.

These are not two systems sharing infrastructure — they are one system
where the primitive library happens to mix rule-based and learned
components. The framing earns the project's name.

## Risks worth pre-committing to as informative

1. **Combinatorial blowup in symbolic search.** Rule-application search
   spaces are notoriously large. The carver's mandate-driven backward
   chaining might tame it (mandates pin intermediate shapes); it might
   not. If carve-fails spike past 50% on Variant A, that's a finding
   about action-principle tractability under symbolic search — and
   suggests the v0 carver budget needs to scale with rule library size.

2. **Pattern-matching cost dominating runtime.** Per carving step the
   carver may run pattern matching against many trees. If matching
   cost dominates, the budget cap becomes a wall-clock cap rather than
   a search-depth cap. Worth instrumenting early with per-step timing.

3. **`EvaluateBinaryOp` collapsing to "always picked" mode.** If
   evaluating literals is always cheaper for the carver than applying
   a symbolic rule, the pure-symbolic primitives (`Distribute`,
   `FactorCommon`, the identity rules) might never get tried. Variant
   A is specifically designed to surface this — its mandate cannot be
   satisfied without symbolic rewriting. If the carver fails Variant A
   with high frequency, the test is unflattering: it would mean the
   selection mechanism still doesn't reach symbolic alternatives even
   with ε-greedy exploration.

4. **Cycle prevention for tree-shaped targets.** The current
   `(target, primClass)` cycle key uses value equality on `target`.
   For tree-shaped values, structural-equality checks could be
   expensive on deep trees. May need a cheap structural-hash check or
   a depth limit on the cycle-detection pass.

## Phased implementation

Each phase produces something runnable to mitigate the §10.6
infrastructure-cost risk, same discipline as v0.

- **Phase 0 — `ParseTreeValue` + parser.** Add the type. Add a
  `ParseExpression` primitive that takes a string and produces a
  parse tree. Hand-wired test: `"(2 + 3) * 4"` parses correctly;
  `ValueDistance` works on trees.
- **Phase 1 — Pattern-matching substrate.** Add `Pattern`, `Matcher`,
  `Bindings`. Standalone test: a few patterns, a few trees, match and
  substitute round-trips.
- **Phase 2 — One pure-symbolic rule (`IdentityZero`).** Wrap it as a
  `Primitive`. Manually constructed `ComputationGraph` applies the
  rule and produces the simplified tree. Mandate verification confirms
  the simplified form appears.
- **Phase 3 — Add `EvaluateBinaryOp` with a learned leaf.** This is
  the heterogeneity primitive. Wrap an MLP (the existing one from v1
  works) inside a rule that pattern-matches on `Lit(_) op Lit(_)`.
  Manually-constructed graph evaluates a small expression. Confirms
  the learned-component-inside-a-rule pattern works.
- **Phase 4 — Carver inverters for tree-shaped values.** Extend
  `BackwardChainingCarver.inferInputs` to dispatch on rewrite
  primitives. Reuse the pattern-matching substrate to invert. Manually
  trigger a carving for a small expression; verify the carved chain
  matches one of the expected orchestrations.
- **Phase 5 — Full primitive library + demo.** Add `Distribute`,
  `FactorCommon`, `IdentityOne`. Run the Variant A and Variant B demo.
  Compare carved structures.

Estimated total scope: ~1000-1500 lines across all phases. Bigger than
v0's substrate (which was ~2000 lines for everything); smaller than
the existing codebase. The biggest unknown isn't lines of code — it's
whether carver search remains tractable at this primitive-library
size.

## What v2 explicitly defers

- **Sub-tree rule application.** v2 applies rules at the root of the
  input tree only. A v3 question is whether the carver should
  automatically search over sub-trees for rule-application sites, or
  whether that should be encoded as additional rules
  (e.g., a `Recurse` rule that descends into a child).
- **Conditional rules / guards.** All v2 rules are unconditional. Real
  symbolic systems often have rules like `?a / ?b → ?a * inv(?b) when ?b ≠ 0`.
  Guards complicate inversion meaningfully; deferred.
- **Equational reasoning beyond rewriting.** No completion procedures,
  no congruence closure, no decision procedures. v2 stays in the
  unidirectional-rewrite setting.
- **Learning to discover rules.** All rules are hand-specified. The
  framework selects among them but does not invent them. Whether the
  action principle could *propose* new rules from observed successful
  carvings is a fascinating but separate question.

## Open question worth flagging upfront

The v0 carver's `(target, primClass)` cycle key was a v0 expedient
to prevent inverter loops between specific primitive pairs (`NumToMat`
and `MatToNum`). For v2's rule library, the analogous concern is more
subtle: applying `Distribute` then `FactorCommon` returns to the same
tree, creating a class-level loop the current key would block; but
applying `Distribute` to one sub-tree and `Distribute` to another is a
legitimate operation the current key might also block.

Two ways to fix:

1. **Switch to `(target, primitive instance)`** — block exact-instance
   loops only.
2. **Switch to per-call sequence of rule applications** — track which
   rules have been applied to which tree, allowing repeated rules
   on different sub-trees but blocking redundant rules on the same
   target.

The former is simpler and probably enough for v2 (since rules apply
at root, and the same target tree won't legitimately need two
applications of the same rule). The latter would be needed if v3
adds sub-tree rule application.

## Bottom line

v2 is ~5 deliverables, ~1000-1500 lines, ~6 phases. It crystallizes
the dual claim by adding the missing piece: a primitive library where
symbolic rules and learned components coexist under the same carving
machinery. The substrate already supports it; v2 just builds the
content.

If the demo passes, the architecture has earned its name. If it
doesn't, we'll have learned something specific about which assumption
fails, and the project's positioning becomes "we built a heterogeneous
network composition platform that *could* host symbolic rewriting if
those assumptions were addressed" — which is still a substantive
claim, just a narrower one.
