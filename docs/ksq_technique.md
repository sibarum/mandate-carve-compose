# KSQ: Key-Split-Quaternion Embedding Architecture

## Core Idea

The model is an embedding table into a fixed algebraic structure. No learned weight matrices. Inputs map to coefficients over a small set of anchor points in the split-quaternion algebra $\mathbb{H}_s$; "computation" is geodesic interpolation in the algebra's unit group $\mathrm{PSL}(2,\mathbb{R})$.

The expressive power lives in the *substrate* (the algebra), not in the parameters. The embedding indexes into structure that's already there.

---

## Substrate

Split-quaternion algebra: basis $\{1, i, j, k\}$ with $i^2 = -1$, $j^2 = +1$, $k^2 = +1$, $ij = k$.

Isomorphic to $M_2(\mathbb{R})$ — implement as $2 \times 2$ real matrices for speed.

**Isomorphism:**

$$a + bi + cj + dk \;\longleftrightarrow\; \begin{pmatrix} a+c & b+d \\ -b+d & a-c \end{pmatrix}$$

All algebraic operations reduce to $2 \times 2$ real matmuls.

---

## Anchor Set

Pick a small fixed set of anchors $\{K_1, \ldots, K_n\}$ in $\mathbb{H}_s$, chosen to cover the subalgebra lattice.

**Minimal viable set (the Möbius cardinals):**

| Anchor | Element | Role | Subalgebra |
|--------|---------|------|------------|
| $K_0$ | $1$ | Identity | Scalar ($\mathbb{R}$) |
| $K_i$ | $i$ | Elliptic / rotation generator | $\mathbb{C}$ |
| $K_j$ | $j$ | Hyperbolic / boost generator | $\mathbb{R}[j] \cong \mathbb{R} \oplus \mathbb{R}$ |
| $K_\infty$ | $\tfrac{1}{2}(1+j)$ | Idempotent / parabolic | Null cone |

Four anchors is the smallest set that touches all qualitatively distinct regimes. Scale by adding more elliptic/hyperbolic rotations or additional idempotents.

**Anchors are fixed, not learned.**

---

## Embedding

Each input token $x$ maps to a coefficient vector $e(x) \in \mathbb{R}^n$ via a learned embedding table. This is the only learned parameter.

For numerical stability, parametrize $e(x)$ on the simplex or sphere (softmax over logits, or unit-norm projection). Coefficients are interpolation weights.

---

## Anchor Combination (the "Operation")

Given coefficients $\alpha = e(x)$, build the corresponding split-quaternion:

$$Q(x) = \exp\!\left(\sum_i \alpha_i \log K_i\right)$$

where $\log$ and $\exp$ are matrix log/exp on the $M_2(\mathbb{R})$ representation. This is geodesic combination in $\mathrm{PSL}(2,\mathbb{R})$.

**Prototyping shortcut:** linear combination $Q(x) = \sum_i \alpha_i K_i$ in $\mathbb{H}_s$, normalized afterward. Loses exactness on the unit group but is differentiable and fast. Start here; upgrade to geodesic later if needed.

---

## Layer / Sequence Operation

A sequence of inputs $x_1, x_2, \ldots, x_T$ becomes a sequence of split-quaternions $Q_1, Q_2, \ldots, Q_T$. The forward pass composes them:

$$S_T = Q_T \cdot Q_{T-1} \cdots Q_1$$

Non-commutative product in $\mathbb{H}_s$ / $M_2(\mathbb{R})$. $S_T$ is the accumulated state — a single $2 \times 2$ real matrix.

For attention-like behavior, compute pairwise products $Q_t \cdot Q_s$ and read scalar invariants (trace, determinant, idempotent components).

---

## Readout

Given final state $S_T$:

1. Compute anchor coefficients $\beta_i$ by projecting onto each $K_i$:
   $$\beta_i = \frac{\mathrm{tr}(S_T K_i^\top)}{\mathrm{tr}(K_i K_i^\top)}$$
   (or solve the linear system $\sum_i \beta_i K_i = S_T$).
2. Pass $\beta$ through an output embedding table to produce logits over the vocabulary / output classes.

The output embedding is the second (and final) learned parameter.

---

## Training

Standard cross-entropy or MSE loss against targets. Autograd flows through $M_2(\mathbb{R})$ matmuls — well-supported in any framework.

**Subalgebra-specialization regularizer:** for each input embedding $e(x)$, add a penalty

$$\lambda \sum_{i \neq j} \alpha_i \alpha_j \cdot d(K_i, K_j)$$

where $d$ is the distance between anchor subalgebras. This pushes embeddings toward single-anchor concentration when the task allows clean specialization.

---

## Inference Specialization

After training, for each embedding $e(x)$:

1. Find the dominant anchor(s).
2. If one anchor dominates → replace $Q(x)$ with $K_{\text{dom}}$ directly (no exp/log needed).
3. If two anchors in the same subalgebra dominate → project $Q(x)$ onto that subalgebra (e.g., drop $j, k$ components if in $\mathbb{C}$).
4. Replace the full $\mathbb{H}_s$ matmul with the specialized subalgebra's cheaper operation:

| Subalgebra | Operation | Cost |
|------------|-----------|------|
| Scalar | Real multiply | 1× |
| $\mathbb{C}$ | Complex multiply | ~2× cheaper |
| $\mathbb{R}[j]$ | Componentwise multiply in idempotent basis $e_\pm$ | ~4× cheaper |
| Full $\mathbb{H}_s$ | $2 \times 2$ real matmul | baseline |

Collapse is **exact** when the embedding is cleanly in one subalgebra; lossy otherwise. The regularizer makes it clean.

---

## Implementation Skeleton

```python
import torch
import torch.nn as nn

# Anchors as 2x2 real matrices (M_2(R) representation of H_s)
ANCHORS = torch.tensor([
    [[1, 0], [0, 1]],            # K_0 = 1 (scalar)
    [[0, 1], [-1, 0]],           # K_i = i (elliptic generator)
    [[1, 0], [0, -1]],           # K_j = j (hyperbolic generator)
    [[0.5, 0.5], [0.5, 0.5]],    # K_inf = (1+j)/2 (idempotent)
], dtype=torch.float32)  # [n_anchors, 2, 2]


class KSQModel(nn.Module):
    def __init__(self, vocab_size, n_anchors=4, out_vocab=None):
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, n_anchors)
        self.output_embed = nn.Linear(n_anchors, out_vocab or vocab_size)
        self.register_buffer('anchors', ANCHORS)

    def to_quaternion(self, x):
        # x: [batch, seq] of token ids
        alpha = torch.softmax(self.input_embed(x), dim=-1)         # [B, T, n]
        Q = torch.einsum('bsn,nij->bsij', alpha, self.anchors)     # [B, T, 2, 2]
        return Q

    def forward(self, x):
        Q = self.to_quaternion(x)             # [B, T, 2, 2]
        S = Q[:, 0]
        for t in range(1, Q.shape[1]):
            S = torch.matmul(S, Q[:, t])      # non-commutative composition
        beta = torch.einsum('bij,nij->bn', S, self.anchors)  # anchor projection
        return self.output_embed(beta)
```

Two embedding tables, fixed anchor buffer, a loop of $2 \times 2$ matmuls. That is the whole model.

---

## Initial Test Tasks

Pick small algebraic problems where the expected specialization is known:

1. **Parity / modular arithmetic** — should specialize to the elliptic subalgebra ($\mathbb{C}$, rotations on $S^1$). Verify by inspecting trained $\alpha$ vectors.
2. **Monotone sequence detection** — should specialize to the hyperbolic subalgebra ($\mathbb{R}[j]$, boosts).
3. **Bracket matching / Dyck language** — should use the idempotent anchor for open/close routing on the null cone.
4. **Mixed task (e.g. modular arithmetic with monotone constraint)** — should retain full $\mathbb{H}_s$ usage; specialization should *fail to collapse*, and the regularizer should reveal this.

If the predicted specializations show up in the trained embeddings, the architecture is doing what the theory says it should.

---

## Why This Works

- **Convex-ish training.** The algebra is fixed; only embeddings are learned. The optimization is closer to matrix factorization with a structured factor than to deep network training. No deep-net saddles from learned multiplicative parameters.
- **Cheap inference.** $2 \times 2$ real matmul per token. Specialization collapses this further per-embedding.
- **Legible interpretability.** A trained embedding *is* a position on the subalgebra lattice. The model's behavior on input $x$ is readable from $e(x)$'s dominant anchor.
- **Unified substrate.** Polynomial (hyperbolic), Fourier (elliptic), and routing/sparsity (null cone) behaviors all live in one algebra. The "two-pathway + mixer" architecture becomes one bilinear operation; the mixer is position on the manifold.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbb{H}_s$ | Split-quaternion algebra |
| $M_2(\mathbb{R})$ | Real $2 \times 2$ matrices (isomorphic to $\mathbb{H}_s$) |
| $\mathrm{PSL}(2,\mathbb{R})$ | Unit group mod center; carries the geodesic structure |
| $K_i$ | Fixed anchor in $\mathbb{H}_s$ |
| $e(x)$ | Learned embedding (anchor coefficients) for input $x$ |
| $\alpha_i$ | Components of $e(x)$ |
| $Q(x)$ | Split-quaternion built from $e(x)$ |
| $S_T$ | Accumulated sequence state |
| $\beta_i$ | Anchor coefficients of $S_T$ at readout |
| $e_\pm$ | Idempotents $\tfrac{1}{2}(1 \pm j)$ |
