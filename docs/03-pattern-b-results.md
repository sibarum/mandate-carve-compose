# STRNN v1 Pattern B — Competitive Coexistence

## What was tested

Both `MlpPrimitive` and `TransformerPrimitive` filling each
`LearnedArithmetic` role (ADD and MUL) in the *same* transformation
graph. The carver picks among them; edge stats accumulate. After
training, primitive-level competition pruning checks whether the
framework can correctly identify and remove an underperforming
implementation.

To make the test diagnostic, MUL pretraining was made intentionally
asymmetric:

| Primitive | Pretraining budget | Single-digit MAE |
|-----------|-------------------|------------------|
| `mlp_add` | 4000 epochs       | 0.085            |
| `tfm_add` | 4000 epochs       | 0.093            |
| `mlp_mul` | 12000 epochs      | 0.411            |
| `tfm_mul` | 2000 epochs (1/6 of mlp's MUL budget) | **4.380** |

If the framework's selection mechanism works, `tfm_mul` should accumulate
lower edge scores and get pruned. ADD primitives should remain
comparable.

## What was added to support the test

Two pieces, in the order they were needed:

1. **`Pruner.prunePrimitiveCompetition(tg)`** — decoupled from the
   existing `prune(tg)`. Grouping is by `(MlpRole)` rather than by edge
   type-pair: all `LearnedArithmetic` primitives sharing a role compete
   with each other. When the best primitive's average outgoing-edge
   score beats a competitor's by at least the margin (and both have at
   least minSamples worth of data), all of the loser's outgoing AND
   incoming edges are pruned.

2. **Epsilon-greedy exploration in `BackwardChainingCarver`** — added
   after the initial run revealed a selection-mechanism issue
   (described below). With probability ε the candidate-ranking step
   skips the score-based sort and uses random order, letting alternative
   edges accumulate samples instead of locking in to whichever edge won
   the initial coin flip. Default ε is 0.0 (preserving v0 behavior);
   Pattern B uses ε=0.10.

The existing edge-pair pruner stays as-is. The two pruning modes serve
different purposes: edge-pair for "which structural slot wins"
(currently too coarse to be useful), primitive-competition for "which
implementation wins at this role".

## Run 1: pure exploitation (ε=0)

Trained 6000 examples with no in-loop pruning, no exploration. Per-primitive
aggregate edge stats:

```
role ADD:
  mlp_add       out=0.601 (n=5115)   in=0.601 (n=4341)
  tfm_add       out=0.500 (n=5)      in=0.500 (n=5)
role MUL:
  mlp_mul       out=0.647 (n=1812)   in=0.647 (n=1812)
  tfm_mul       out=0.600 (n=4494)   in=0.600 (n=4494)
```

`prunePrimitiveCompetition` pruned exactly `tfm_mul`'s 5+5 edges. The
framework correctly identified the weakest implementation.

But the per-edge breakdown revealed something unexpected:

```
slot                         winner                loser (0 samples)
─────────────────────────────────────────────────────────────────────
compose ← matrix(9)          mlp_mul → compose     tfm_mul → compose
                             1812 samples          0 samples

mat_to_num ← matrix(9)       tfm_mul → mat_to_num  mlp_mul → mat_to_num
                             4494 samples          0 samples

ADD slots                    mlp_add wins          tfm_add never tried
                             ~5000 samples         5 samples
```

`tfm_mul` was used *more* than `mlp_mul` overall (4494 vs 1812) despite
being demonstrably worse. Aggregate primitive scores still distinguished
them correctly, but specific edges had locked in: whichever alternative
got randomly picked first for a given slot stayed picked indefinitely.
The unpicked alternative remained at score 0.5 forever and never got
tried.

This was a real selection-mechanism failure, not a quality issue. The
carver's pure-exploitation default meant edge stats reflected *which
edge got lucky early*, not *which primitive was better at that slot*.

## Run 2: ε-greedy exploration (ε=0.10)

Same setup, ε=0.10. Per-primitive aggregate edge stats:

```
role ADD:
  mlp_add       out=0.609 (n=4230)   in=0.615 (n=4153)
  tfm_add       out=0.611 (n=526)    in=0.611 (n=516)
role MUL:
  mlp_mul       out=0.627 (n=2762)   in=0.612 (n=2663)
  tfm_mul       out=0.567 (n=712)    in=0.567 (n=708)
```

Per-edge stats now show every alternative getting non-zero samples:

```
edge                         score    samples
mlp_mul → mat_to_num          0.601    2302   (was 0)
tfm_mul → mat_to_num          0.568     333   (was 4494)
mlp_mul → compose             0.653     460
tfm_mul → compose             0.567     708
mlp_add → mat_to_num          0.614    4118
tfm_add → mat_to_num          0.606     282
mlp_add → compose             0.603     112
tfm_add → compose             0.616     244
```

Three predictions verified:

| Prediction                                                       | Verdict |
|------------------------------------------------------------------|--------|
| `mlp_mul → mat_to_num` accumulates non-zero samples              | ✓ 2302 (from 0) |
| Both alternatives get meaningful samples at each slot            | ✓ Lowest is 244 |
| `mlp_mul → mat_to_num` scores higher than `tfm_mul → mat_to_num` | ✓ 0.601 vs 0.568 |

Selection now reflects quality. `mlp_mul` total samples ≈ 2762,
`tfm_mul` ≈ 712 — the better primitive is selected ~4× more often
(reversed from the locked-in 1:2.5 ratio in run 1). `tfm_add` finally
got exercised (526 samples vs 5) and scored statistically
indistinguishably from `mlp_add` (0.611 vs 0.609), correctly reflecting
that they're equally good at addition.

`prunePrimitiveCompetition` still cleanly identified `tfm_mul` as the
loser, with the score gap now widened to 0.060 (well above the 0.03
margin). The pruner did not fire on the ADD role — also correct, since
`mlp_add` and `tfm_add` are genuinely comparable.

Training quality did not regress: average terminal score 0.608 → 0.611,
all-mandates rate 0.048 → 0.058. The 10% exploration cost was paid back
by better selection.

## What Pattern B licenses

- **The action principle's evaluation signal is meaningful at the
  primitive level.** Across many examples, edge scores reflect a
  primitive's average contribution to mandate satisfaction.
- **Primitive-level pruning works.** Given a configurable margin and
  sample threshold, the framework identifies and removes a
  consistently-worse primitive without disturbing the rest of the
  transformation graph.
- **Selection is quality-responsive when exploration is enabled.** With
  ε-greedy at a modest 10%, alternative edges accumulate samples,
  per-edge scores reflect quality, and selection probabilities track
  edge scores in the long run.
- **The §10.2 cross-graph credit aggregation question has a partial
  answer:** the per-edge running-mean update propagates terminal score
  evenly to every traced edge in a carving. This is a conservative
  aggregation rule (every edge in a successful carving gets credit;
  every edge in a failed one gets blame). It works well enough to
  surface the weakest competitor when the gap is real *and* exploration
  exposes alternatives.

## What Pattern B does NOT demonstrate

- **Pure exploitation is the wrong default.** The pure-exploitation
  default produced misleading results in run 1. v0 and Pattern A
  worked fine without exploration only because there was nothing
  competing. Once primitives compete, exploration is necessary. ε-greedy
  is plumbed in but the right value is task-dependent (we tested only
  ε=0.10).
- **The competition was deliberately asymmetric.** With two primitives
  pretrained equally, would the score gap exceed the 0.03 margin in
  6000 steps? Run 2's ADD-role behavior (gap < 0.01, no pruning) is
  weak evidence that the framework correctly does NOT fire when
  primitives are genuinely comparable, but it's only one symmetric
  comparison.
- **Pruning happened post-hoc.** The demo deliberately disables in-loop
  pruning and runs `prunePrimitiveCompetition` once at the end. Whether
  pruning during training would help or hurt convergence is unanswered.

## Open questions surfaced by Pattern B

1. **Should ε anneal over training?** Constant ε=0.10 worked here.
   Standard bandit practice is to decay (more exploration early, more
   exploitation later). For longer training runs or noisier signals,
   annealing might help.

2. **Should cycle-prevention be primitive-instance-aware instead of
   class-aware?** The current `(target, primClass)` cycle key prevents
   the same-class primitive from filling repeated slots. With
   exploration breaking lock-in, this matters less, but it still biases
   selection in subtle ways. A `(target, primitive identity)` key would
   prevent exact-instance loops while letting any other primitive (same
   class or different) fill alternative slots. Worth revisiting
   alongside the symbolic-rewrite work, where rule primitives may
   legitimately be applied multiple times to different sub-trees.

3. **Could symmetric Pattern B detect equivalence?** Run 2's ADD-role
   behavior is suggestive (no pruning fired, scores within 0.002), but
   that's one data point. A dedicated symmetric run with multiple
   architectures pretrained equally would test the framework's
   discrimination threshold from the other side.

## Bottom line

Pattern B passes for its stated test: the framework correctly identified
and pruned the deliberately-undertrained primitive. The first run also
surfaced a real selection-mechanism failure (per-edge lock-in under pure
exploitation); the second run, with ε-greedy exploration enabled,
demonstrated that the failure was specific to exploitation-only
selection and is fixable with a small, well-understood mechanism.

This unblocks the path to richer compositions: with selection actually
responsive to evaluation, putting rule-based primitives next to
learned-leaf primitives in the same library becomes a meaningful test
of the framework's discrimination — the carver can actually pick rules
over evaluations (or vice versa) based on which path the action
principle credits more. That is the v2 work.
