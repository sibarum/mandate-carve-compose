# KSQ: Key-Split-Quaternion Embedding Architecture

## Core Idea

The model is an embedding table into a fixed algebraic structure. There are no learned weight matrices in the conventional sense — only embedding tables. Inputs map to **signed** coefficients over a small set of anchor points in the split-quaternion algebra $\mathbb{H}_s$. A **single bilinear step** in the algebra produces the model's nonlinearity. A linear readout maps the result to outputs.

The expressive power lives in the substrate (the algebra), not in stacked parameters. The embedding indexes into structure that is already there; the bilinear step is the one multiplicative operation the architecture is permitted.

---

## Substrate

Split-quaternion algebra: basis $\{1, i, j, k\}$ with $i^2 = -1$, $j^2 = +1$, $k^2 = +1$, $ij = k$.

Isomorphic to $M_2(\mathbb{R})$ — implement as $2 \times 2$ real matrices for speed and simple autograd.

**Isomorphism:**

$$a + bi + cj + dk \;\longleftrightarrow\; \begin{pmatrix} a+c & b+d \\ -b+d & a-c \end{pmatrix}$$

All algebraic operations reduce to $2 \times 2$ real matmuls. The bilinear step is one such matmul.

---

## Anchor Set

Pick a small fixed set of anchors $\{K_1, \ldots, K_n\} \subset \mathbb{H}_s$ chosen to cover the subalgebra lattice.

**Minimal viable set (the Möbius cardinals):**

| Anchor | Element | Role | Subalgebra |
|--------|---------|------|------------|
| $K_0$ | $1$ | Identity | Scalar ($\mathbb{R}$) |
| $K_i$ | $i$ | Elliptic / rotation generator | $\mathbb{C}$ |
| $K_j$ | $j$ | Hyperbolic / boost generator | $\mathbb{R}[j] \cong \mathbb{R} \oplus \mathbb{R}$ |
| $K_\infty$ | $\tfrac{1}{2}(1+j)$ | Idempotent / parabolic | Null cone |

Four anchors is the smallest set that touches all qualitatively distinct regimes. Scale by adding rotations or further idempotents if the task demands finer structure.

**Anchors are fixed at construction; only the embeddings into anchor-coefficient space are learned.**

---

## Embedding (Signed Coefficients)

Each input token $x$ maps to a coefficient vector $\alpha(x) \in \mathbb{R}^n$ via a learned embedding table followed by a **tanh** activation:

$$\alpha(x) = \tanh\bigl(E[x]\bigr)$$

where $E$ is the learned embedding table.

**Why tanh, not softmax.** A softmax constrains $\alpha$ to the probability simplex (nonneg, sums to 1). This destroys the algebra's sign structure: anchors whose squares are negative (like $K_i^2 = -1$) require *signed* coefficients to produce discriminative bilinear outputs. Under softmax, the elliptic ($K_i$) basin is unreachable; under tanh, the elliptic and hyperbolic subalgebras are symmetric basins of equal accessibility.

The natural domain for anchor coefficients is the Lie algebra of $\mathrm{PSL}(2,\mathbb{R})$ (or its enveloping algebra), which is unconstrained reals — not the simplex. Tanh is a soft bounded approximation; unconstrained reals with weight decay also work.

---

## Sequence Pooling

The architecture is **single-layer in the algebra**: there is no chain of matmuls along the sequence. Sequence structure is handled at the embedding level, by pooling token coefficients into a single sequence-level coefficient vector before the bilinear step.

For a sequence $x_1, \ldots, x_T$:

$$\alpha_{\text{seq}} = \tanh\!\left(\sum_t E[x_t]\right)$$

Sum-pool the logits, then apply tanh once. This is the only place sequence position is collapsed, and it happens *before* the algebra is touched. Other pooling functions (mean, max, attention-weighted) are admissible; the constraint is that pooling produces a single $\alpha \in \mathbb{R}^n$ to feed into the algebra.

---

## Lifting into the Algebra

Build the split-quaternion corresponding to the pooled coefficients:

$$Q = \sum_i \alpha_i K_i$$

Linear combination in $\mathbb{H}_s$ (equivalently, in $M_2(\mathbb{R})$). This is the "embedding into the structured space" — $Q$ is the single $2 \times 2$ real matrix that represents the input.

**Optional upgrade:** geodesic combination $Q = \exp\!\left(\sum_i \alpha_i \log K_i\right)$ on the Lie group $\mathrm{PSL}(2,\mathbb{R})$. More principled but harder to differentiate and not necessary for the architecture to function. Start with the linear combination.

---

## The Single Bilinear Step

One multiplication in the algebra:

$$S = Q \cdot Q$$

This is the entire nonlinearity. It is one $2 \times 2$ real matmul. Variants are admissible — $S = Q \cdot M$ for a fixed reference $M$, or $S = Q_1 \cdot Q_2$ if the architecture splits the input into two streams — but the canonical form is $Q^2$, the squaring step.

This is the multiplicative operation that buys "the entire space of quadratic forms" with zero learned parameters in the nonlinearity itself. The algebra's structure — non-commutativity, indefinite signature, idempotents, zero-divisors — is fully expressed in this one step.

---

## Readout

Project $S$ onto each anchor via the Frobenius inner product:

$$\beta_i = \frac{\langle S, K_i \rangle_F}{\|K_i\|_F^2} = \frac{\mathrm{tr}(S K_i^\top)}{\mathrm{tr}(K_i K_i^\top)}$$

This produces a coefficient vector $\beta \in \mathbb{R}^n$ that describes "where $S$ lies in anchor-coordinate space."

A learned linear layer then maps $\beta$ to output logits:

$$\text{logits} = W \beta + b$$

The output linear layer is the second (and final) set of learned parameters. For classification, apply softmax to logits and use cross-entropy loss.

**Readout note.** The readout-anchor pairing controls which subalgebras are discriminative for which tasks. Different readouts (e.g., reading off multiple powers of each anchor, or pairwise products $K_i K_j$) expose different aspects of $S$ and admit different specialization patterns. The Frobenius-against-anchors readout is the minimal choice; richer readouts are admissible and may be necessary for tasks where the minimal one is degenerate.

---

## Training Objective

Standard task loss (cross-entropy for classification, MSE for regression) on the readout logits. Autograd flows through $M_2(\mathbb{R})$ matmuls, tanh, and the bilinear step. The bilinear backward for $S = Q^2$ is:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot Q^\top + Q^\top \cdot \frac{\partial \mathcal{L}}{\partial S}$$

— the symmetric chain through the non-commutative squaring.

---

## Specialization Regularizer

Add a penalty that encourages each embedding to commit to a single anchor (or a small set), preventing diffuse mixtures that don't cleanly inhabit any subalgebra:

$$\mathcal{R}_{\text{spec}} = \lambda \sum_{i \neq j} \alpha_i^2 \alpha_j^2$$

**Sign-safe form.** This is the squared-product form, even in each $\alpha$. It is zero when $\alpha$ is axis-aligned (single anchor active) and positive otherwise, regardless of sign. The linear form $\sum_{i\neq j} \alpha_i \alpha_j$ is incompatible with signed coefficients (it rewards mixed-sign mixing rather than penalizing it); the squared-product form has the right structure under tanh.

**Typical magnitude:** $\lambda \in [0.01, 0.1]$ is enough to sharpen specialization without disrupting solve rate. Larger $\lambda$ risks forcing premature commitment.

---

## Anti-Degeneracy Regularizer (When Needed)

The bilinear product $Q^2$ has algebraic zeros at certain corner configurations. Most notably, joint saturation $\alpha_i = -1, \alpha_j = +1$ produces $Q^2 = 0$ because $K_i K_j + K_j K_i = 0$ (the anti-commutator vanishes for orthogonal-signature generators).

When tanh saturation drives $\alpha$ to such corners, $S$ collapses to zero and the readout loses all signal. The specialization regularizer above already deflects from these corners as a side effect (penalizing simultaneous saturation of two anchors), so an explicit anti-saturation penalty is usually unnecessary. If finer control is needed:

$$\mathcal{R}_{\text{sat}} = \mu \sum_i \alpha_i^{2k}$$

for $k \geq 2$ penalizes saturation directly without affecting moderate-magnitude coefficients.

---

## Cross-Vocabulary Regularizer (For Discrimination)

The per-token specialization regularizer is myopic — it has no incentive to push *different* tokens toward *different* anchors. For tasks requiring token-level role differentiation, add a contrastive cross-vocab term:

$$\mathcal{R}_{\text{cross}} = \nu \sum_{x \neq y} \langle \alpha(x), \alpha(y) \rangle^2$$

Penalizes pairs of tokens whose coefficient vectors are aligned. Encourages the embedding table's rows to be near-orthogonal, ensuring distinct tokens commit to distinct anchors.

This is the standard decorrelation regularizer from contrastive representation learning, repurposed for anchor specialization.

---

## Inference Specialization

After training, for each token $x$:

1. Compute $\alpha(x)$ and find the dominant anchor(s) by $\arg\max_i |\alpha_i(x)|$.
2. If one anchor dominates → replace $Q(x)$ with $\mathrm{sign}(\alpha_{\text{dom}}) \cdot K_{\text{dom}}$ directly.
3. If two anchors in the same subalgebra dominate → project onto that subalgebra (e.g., drop $j, k$ components if in $\mathbb{C}$).
4. Replace the bilinear step with the specialized subalgebra's cheaper operation:

| Subalgebra | Operation | Cost |
|------------|-----------|------|
| Scalar | Real multiply | 1× |
| $\mathbb{C}$ | Complex multiply | ~2× cheaper than $\mathbb{H}_s$ |
| $\mathbb{R}[j]$ | Componentwise multiply in idempotent basis $e_\pm$ | ~4× cheaper |
| Full $\mathbb{H}_s$ | $2 \times 2$ real matmul | baseline |

The collapse is **exact** when the embedding is cleanly in one subalgebra. The specialization regularizer makes this the typical case.

---

## Algorithm Summary

```
Input: token sequence x_1, ..., x_T
Parameters: embedding table E ∈ R^(V × n), readout W ∈ R^(out × n), bias b ∈ R^out
Fixed: anchors K_1, ..., K_n ∈ M_2(R)

# Embedding and pooling
sum_logits = Σ_t E[x_t]
α = tanh(sum_logits)                          # signed coefficients in R^n

# Lift into algebra
Q = Σ_i α_i K_i                                # single 2x2 real matrix

# Single bilinear step
S = Q · Q                                      # the entire nonlinearity

# Readout
β_i = ⟨S, K_i⟩_F / ‖K_i‖_F²                    # anchor projection
logits = W · β + b

# Loss
L_task   = CrossEntropy(logits, target)
L_spec   = λ · Σ_{i≠j} α_i² α_j²               # per-vocab specialization
L_cross  = ν · Σ_{x≠y} ⟨α(x), α(y)⟩²           # cross-vocab decorrelation (optional)
L        = L_task + L_spec + L_cross
```

---

## Implementation Skeleton

```python
import torch
import torch.nn as nn

# Möbius cardinal anchors as 2x2 real matrices (M_2(R) rep of H_s)
ANCHORS = torch.tensor([
    [[1, 0], [0, 1]],            # K_0 = 1 (scalar)
    [[0, 1], [-1, 0]],           # K_i (elliptic generator)
    [[1, 0], [0, -1]],           # K_j (hyperbolic generator)
    [[0.5, 0.5], [0.5, 0.5]],    # K_inf = (1+j)/2 (idempotent)
], dtype=torch.float32)


class KSQModel(nn.Module):
    def __init__(self, vocab_size, n_anchors=4, out_dim=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_anchors)
        self.readout = nn.Linear(n_anchors, out_dim or vocab_size)
        self.register_buffer('anchors', ANCHORS)
        self.register_buffer(
            'anchor_norms_sq',
            torch.einsum('nij,nij->n', ANCHORS, ANCHORS)
        )

    def forward(self, x):
        # x: [B, T] token ids

        # 1. Embed and pool to single α
        logits = self.embed(x).sum(dim=1)           # [B, n]
        alpha = torch.tanh(logits)                  # [B, n], signed

        # 2. Lift into algebra
        Q = torch.einsum('bn,nij->bij', alpha, self.anchors)  # [B, 2, 2]

        # 3. Single bilinear step
        S = torch.matmul(Q, Q)                      # [B, 2, 2]

        # 4. Readout
        beta = torch.einsum('bij,nij->bn', S, self.anchors) / self.anchor_norms_sq
        return self.readout(beta), alpha            # return α for regularizer

    @staticmethod
    def specialization_loss(alpha, lam=0.1):
        # Σ_{i≠j} α_i² α_j² = (Σ α_i²)² - Σ α_i⁴
        sq = alpha.pow(2)
        return lam * (sq.sum(dim=-1).pow(2) - sq.pow(2).sum(dim=-1)).mean()
```

The full model: two learned tensors (embedding table + readout linear), one fixed anchor buffer, and one $2 \times 2$ matmul per input as the entire nonlinear computation.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbb{H}_s$ | Split-quaternion algebra |
| $M_2(\mathbb{R})$ | Real $2 \times 2$ matrices (isomorphic to $\mathbb{H}_s$) |
| $\mathrm{PSL}(2,\mathbb{R})$ | Unit group mod center; carries the geodesic structure |
| $K_i$ | Fixed anchor in $\mathbb{H}_s$ |
| $\alpha(x)$ | Signed anchor coefficients for input $x$, post-tanh |
| $Q$ | Split-quaternion built from $\alpha$ (linear combination of anchors) |
| $S$ | Bilinear step output, $S = Q \cdot Q$ |
| $\beta$ | Anchor coefficients of $S$ at readout |
| $e_\pm$ | Idempotents $\tfrac{1}{2}(1 \pm j)$, basis for $\mathbb{R}[j]$ specialization |

---

## Design Principles (Recap)

1. **Substrate over parameters.** The algebra is fixed and carries the structure; embeddings only index into it.
2. **One bilinear step.** A single squaring buys the full quadratic function class; the architecture is shallow in the algebra.
3. **Signed coefficients.** Tanh, not softmax. The algebra's sign structure must be preserved in the parametrization.
4. **Pool at the embedding level.** Sequence structure is collapsed before the algebra is touched; no chained matmuls.
5. **Specialization is post-hoc and exact.** Trained embeddings classify into subalgebras; inference replaces the full operation with the specialized cheaper one without approximation.
6. **Regularizers respect the algebra.** Specialization and decorrelation terms are written in forms compatible with signed coefficients and the algebra's structure (squared products, inner products, not raw linear terms).
