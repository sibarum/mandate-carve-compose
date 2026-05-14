# KSQP raw run data

Per-seed trajectories, loss curves, event logs, and summaries from the
KSQP experiments (iter 7B in `docs/16-ksq-substrate.md`). Each
subdirectory holds the artifacts from one run of `KsqpXorDemo`,
`KsqpKvXorDemo`, or `KsqpKvCircleDemo`; the demo writes the CSVs to
`ksqp-data/<runTag>/` where `runTag` is the demo's first argument.

## Runs

| Subdir | Demo | Task | Notes |
|---|---|---|---|
| `proper-arch/` | `KsqpXorDemo` | T=2 XOR (indexed) | Early run; 0/10 stuck — predates the split-quat-product cross-token aggregation |
| `split-quat-product/` | `KsqpXorDemo` | T=2 XOR (indexed) | 9/10 solved; split-quat-product aggregation in place |
| `original-mapping/` | `KsqpXorDemo` | T=2 XOR (indexed) | 7/10; sign-flip → degree mapping: $+\to-$ ⇒ $p{+}{+}$ |
| `flipped-mapping/` | `KsqpXorDemo` | T=2 XOR (indexed) | 9/10; mapping flipped to $+\to-$ ⇒ $p{-}{-}$ (small A/B improvement) |
| `kv-xor-2d/` | `KsqpKvXorDemo` | 2D continuous XOR, M=4 corners, frozen keys | 10/10 |
| `kv-xor-2d-trainable-keys/` | `KsqpKvXorDemo` | 2D continuous XOR, M=4 corners, trainable keys | 10/10 |
| `kv-circle-8/` | `KsqpKvCircleDemo` | 8 alternating-label clusters on unit circle, trainable keys | 0/10 stuck — scale-up failure case |

## File layout per run

All runs write `summary.csv`, `trajectories.csv`, `events.csv`,
`loss.csv`. KV runs additionally write `attention_samples.csv` and
`stored_keys.csv`; the circle run also writes `clusters.csv`.

- `summary.csv` — one row per seed: final accuracy, CE loss, number of
  events fired, final $p$ per entry, final $\|q\|$ or norm per entry,
  outcome label (`solved` vs `stuck`).
- `trajectories.csv` — epoch-keyed log of per-entry split-quaternion
  parameters and split-quat norm $N(q)$. Used to inspect the path
  through SQ space and the angle of approach to null-cone events.
- `events.csv` — one row per null-cone event: epoch, entry id,
  $p_{\text{before}} \to p_{\text{after}}$, norm just before, norm
  just after re-init, the SQ vector at the event.
- `loss.csv` — per-epoch CE.
- `attention_samples.csv` (KV only) — softmax weights over the M
  stored entries for a fixed sample of queries, at several epochs.
- `stored_keys.csv` (KV only) — final stored-key positions (relevant
  when keys are trainable).
- `clusters.csv` (circle only) — the target cluster centers and
  labels for that run.

Demos overwrite their target directory; pass a unique `runTag`
argument to keep multiple runs around. The CSV column orderings are
defined by the demo source, not by a separate schema file.

## Where to read about results

`docs/16-ksq-substrate.md` (iter 7 §B) is the consolidated narrative.
`docs/KSQP.md` is the original plan with a status block at the top.
