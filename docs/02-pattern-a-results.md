# STRNN v1 Pattern A — Drop-in Architecture Swap

## What was tested

The §9.6 ablation #4 from the design doc, framed in `01-first-phase-results.md`
as v1 question 3:

> Swap one primitive for another of the same type signature. Replace MLP
> nodes with small transformer blocks. If the carved structure transfers
> (the orchestration is the same, only the underlying learner changed),
> this directly demonstrates the framework's component-agnostic nature.

The test compares two transformation graphs that differ only in which
learned component fills the `LearnedArithmetic` role:

| Config | ADD primitive | MUL primitive |
|--------|---------------|---------------|
| MLP    | `Mlp [2, 32, 1]`            | `Mlp [2, 128, 64, 1]`        |
| TFM    | `Transformer (dM=16, dF=64)` | `Transformer (dM=32, dF=128)` |

Both were pretrained on single-digit add/mul to comparable accuracy. Both
were then plugged into the same Phase 6 ablation (50 random `a±b·c` /
`a·b±c` examples; full §9.2 mandates vs. result-only).

## Refactor that made the test possible

Three small changes to the v0 codebase:

1. New interface `LearnedArithmetic extends Primitive { MlpRole role(); }`.
2. `MlpPrimitive` and `TransformerPrimitive` both implement it.
3. `BackwardChainingCarver.inferInputs` matches against `LearnedArithmetic`
   instead of `MlpPrimitive` directly.
4. `Trainable.trainableIdentity()` added so the trainer can dedup `step()`
   calls per-network without knowing the network's class.

Concrete, mechanical, and testable: `CarvingDemo` still passes after the
refactor. No carver behavioral changes.

## Result

| Metric                          | MLP        | Transformer |
|---------------------------------|------------|-------------|
| A: success / 50                 | 48         | 48          |
| A: avg nodes per carved graph   | 16.6       | 17.4        |
| A: % of carvings with intermediate product on result path | 67%       | 52%         |
| B (result-only): success / 50   | 0          | 0           |

Primitive-class counts per A-carving (rounded):

| Primitive          | MLP  | Transformer |
|--------------------|------|-------------|
| `LearnedArithmetic` | 2.10 | 2.27       |
| `ComposeMatrices`   | 2.08 | 2.25       |
| `NumberToMatrix`    | 2.58 | 2.77       |
| `MatrixToNumber`    | 1.69 | 1.73       |
| `ParseNumber`       | 2.06 | 2.17       |
| `TokenAt`           | 3.06 | 3.17       |
| `SplitStringAt`     | 2.00 | 2.00       |
| `OutputPrimitive`   | 1.00 | 1.00       |

All within ~0.3 of each other across every primitive class. Identical
structural composition: the orchestration the carver discovers is the same
shape regardless of whether the trainable slots are MLPs or transformers.

**Pattern A passes:** the carved orchestration is component-agnostic to the
extent the v0 demo can test. The framework's "swap one primitive for
another with the same type signature, structure transfers" claim survives
this swap.

## Unexpected finding worth flagging

DOT-diffing the per-example carvings revealed the carver does **not**
strongly distinguish `ADD` from `MUL` when the available numeric anchor
pool admits both as solutions. Concrete example for input `"8+9*1"`
(intermediate product = 9, result = 17):

- The carver can satisfy intermediate_product=9 by either:
  - `MUL(9, 1) = 9` — anchors 9 and 1 are present
  - `ADD(8, 1) = 9` — anchors 8 and 1 are present
- Either choice produces a valid carving. Which one wins depends on the
  shuffle order of candidates inside `inferInputs`.

This is why the per-example carvings between MLP and TFM configs sometimes
look semantically different ("the MLP run used MUL here, the transformer
run used ADD") even though the structural counts are identical at
aggregate. The role tag is **advisory**, not **binding**: the carver
respects type compatibility but not semantic alignment with the input
expression's actual operators.

This wasn't a deliberate design choice — it's a v0 limitation that
Pattern A surfaced. Two ways to look at it:

- **Generous reading**: this is exactly what the design doc means by
  "the same primitive can be used in multiple computation-graph contexts
  with different neighborhoods" (§2.1). Multiple valid orchestrations
  exist; the carver finds one of them.
- **Skeptical reading**: an "intermediate_product" mandate ought to
  constrain to multiplication, not just to "anything-that-yields-9". The
  current carver doesn't enforce that constraint. A v2 mandate could
  carry an operator-role hint, or the carver could prefer role matches.

For Pattern A's purpose, this looseness is in fact what makes the swap
test pass cleanly: the carver uses whatever learned component yields the
mandate value, agnostic to architecture.

## What Pattern A licenses

The component-agnosticism claim from §1.2 / §8 survives a real swap test.
This is a stronger result than v0's interpretability claim because it
exercises the framework's principal application (heterogeneous
composition) in its smallest non-trivial form.

To be precise about what was actually demonstrated:

- One learned component type (MLP) can be replaced with another (a
  transformer) at the same role and signature without changing the
  carved orchestration's shape, success rate, or structural composition.
- The refactor needed to support this was a 4-line interface, not a
  rearchitecture. The carver's coupling to `MlpPrimitive` was incidental,
  not load-bearing.
- The §6.2 search-decomposition payoff (B fails 50/50 under both configs)
  reproduces under the swap — that finding was not architecture-specific.

## What Pattern A does NOT demonstrate

- **Two architectures cooperating in one carving.** Both configs use one
  family at both role slots (all MLPs, or all transformers). True
  heterogeneity (transformer-add + MLP-mul, or different families at
  competing slots) is the next test. Pattern B in the v1 plan.
- **Genuinely different type signatures.** Both `MlpPrimitive` and
  `TransformerPrimitive` have signature `MATRIX(2) → MATRIX(1)`. A
  transformer with signature `TOKEN_LIST → MATRIX` (Pattern C) would
  test heterogeneous *interfaces*, not just heterogeneous *internals*.

## Open questions surfaced by Pattern A

1. **Should role tags be binding?** The advisory-not-binding behavior
   means the carver is more flexible than the design doc may have
   intended. Whether to tighten this depends on whether you want
   mandates to constrain *which operator produced the value* or only
   *that the value was produced*.

2. **Where does role-information actually come from?** Currently the
   role tag is set at TransformationGraph build time (the engineer
   labels the MLP `MUL`). There is no signal connecting that label to
   the input expression's actual operators. A natural v2 question:
   could mandate values carry implicit operator information (e.g.
   "intermediate_product = 9 produced by multiplying"), and could the
   carver prefer role-matching paths?

3. **Can Pattern A scale to mixed configurations?** Pattern B (the v1
   plan's competitive coexistence) puts both MLP and transformer in the
   same transformation graph at the same role and lets edge stats pick.
   This tests the action principle's selection mechanism in a way v0
   couldn't (since v0 had no real alternatives competing).

## Bottom line

Pattern A passes cleanly. The orchestration is structurally identical
across MLP and Transformer configurations, the refactor cost was minimal,
and the v0 §6.2 finding survives the architecture change. The test's
sharpness was actually limited by an unintended carver behavior (advisory
role tags) — surfacing that is itself a useful side effect.

Pattern B is the natural next move: put both architectures in the same
graph at the same role, see if edge stats pick a winner, see if pruning
can finally fire on a meaningful divergence. That experiment would
exercise the action principle's selection machinery for the first time.
