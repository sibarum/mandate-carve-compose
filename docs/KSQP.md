# KSQ with Discrete Degree Control

## Status

Implemented as `sibarum.strnn.ksqp` (`KsqpModel` for indexed lookup,
`KsqpKvModel` for soft-attention KV). Gradient check passes at
machine epsilon. Empirical results in `ksqp-data/`:

- T=2 XOR (indexed, split-quat-product aggregation): **9/10 solved**
  with the implemented sign-to-degree mapping (`split-quat-product/`
  and `flipped-mapping/`); 7/10 with the alternate mapping
  (`original-mapping/`). Earlier `proper-arch/` run is 0/10 from
  before the split-quat-product aggregation was in place.
- 2D continuous XOR via KV cache (M=4 prototypes): **10/10 solved**
  with both frozen and trainable stored keys (`kv-xor-2d/`,
  `kv-xor-2d-trainable-keys/`).
- 8-cluster alternating-label circle (M=8 KV cache): **0/10 stuck**
  (`kv-circle-8/`) — failure mode under investigation.

On the XOR variants the discrete-event mechanism mostly *doesn't*
fire (most solved seeds stay at $p=1$). Iter-5's degree-2
expressivity is enough for XOR; the mechanism's contribution on
higher-degree tasks is not yet demonstrated. See iter-7 §B in
`16-ksq-substrate.md` for the consolidated reading.

Open items below remain open.

## Background

**KSQ** (Keyed Split Quaternion) uses a key-value lookup to retrieve a 4-dimensional parameter vector, which parameterizes a split quaternion transformation applied to the input. The 4 parameters effectively select among the conjugacy classes of the split quaternion group (hyperbolic, parabolic, elliptic, projective), giving the layer the structural variety of a small transformer block in a single algebraic operation.

The known limitation: a single split quaternion application produces a degree-2 transformation. Higher-degree relationships are not directly representable, and parameter configurations near the null cone produce information-destroying idempotents rather than useful transformations.

## Proposal

Extend the parameter vector from 4 to 5 dimensions. The 5th parameter is a **discrete integer-valued degree selector** that controls a polynomial lift of the input prior to the split quaternion operation. The input is lifted to its degree-d Pi-net representation (the vector of all degree-d monomials in the input components, projected to 4 dimensions), and the split quaternion acts on the lifted vector.

This gives the layer expressivity across a range of polynomial degrees while preserving the algebraic structure of the split quaternion operation. The cost is that the 5th parameter is not gradient-trainable — it is updated by a runtime control mechanism described below.

## Runtime Control of the Degree Parameter

The 5th parameter is updated by a hybrid dynamical system bolted onto the gradient-trained portion of the network. The update rule is triggered by null-cone crossings of the 4 split quaternion parameters.

### Trigger condition

A null-cone event fires when the split quaternion parameter vector crosses (or approaches within threshold) one of the two null-cone sheets. The two sheets correspond to the two idempotents of the split quaternion algebra — each idempotent destroys a distinct information channel.

The mapping from sheet to degree direction depends on which information channel is destroyed:

- Sheet destroying high-degree-content idempotent → **decrement** the 5th parameter
- Sheet destroying low-degree-content idempotent → **increment** the 5th parameter

(The exact mapping depends on how the Pi-net lift orders degrees in the 4-vector passed to the split quaternion. To be determined per implementation.)

### Hysteresis

Because the trigger is a thresholded condition on a continuous quantity, naive implementation will produce instability: parameters near the null cone will flip the 5th parameter back and forth. Hysteresis is required — different thresholds for entering vs. exiting the null-cone region — so that crossings are committed events rather than oscillations.

### Gradient-directed teleport on the 4 split quaternion parameters

When a null-cone event fires, the 4 split quaternion parameters are not reset to identity or left in place. Instead, they are extrapolated through the null cone in the direction of the loss gradient computed just before the event:

```
new_params = pre_event_params + step_size * gradient_direction
```

The step size is chosen large enough to clear the null cone past the exit-hysteresis threshold, landing firmly inside whichever conjugacy region the gradient was already pushing toward.

### Caveats on the gradient direction

Gradients near the null cone are ill-conditioned because the split quaternion operation itself becomes singular there. Magnitudes may be large or noisy. The teleport implementation should clip or smooth the gradient before using it as a direction, and the step size may need to be a scheduled or adaptive hyperparameter rather than a fixed constant.

### Coordination between region and degree

The teleport produces a coordinated jump: the 5th parameter changes discretely (degree shifts up or down), and the 4 split quaternion parameters jump across the null cone to a new conjugacy region selected by the gradient.

The 8 possible (target region, degree direction) combinations should be enumerated and checked for consistency. Some pairings may be natural; others may be rare or pathological in practice and worth special-casing.

## Interpretation

The null cone functions as a phase boundary between distinct dynamical regimes of the layer. Crossings are treated as first-order transitions: the order parameter (polynomial degree) changes discretely while the conjugate field (split quaternion parameters) extrapolates smoothly through the boundary.

The resulting system is a piecewise-smooth hybrid dynamical system: continuous gradient flow on the 4 split quaternion parameters within each region, with discrete coordinated jumps at null-cone crossings. Existing literature on hybrid systems, switched systems, and piecewise-smooth dynamical systems should be applicable for stability and convergence analysis.

## Open Items

- Concrete formula for the sheet-to-degree-direction mapping, dependent on Pi-net lift ordering
- Hysteresis thresholds (entry vs. exit)
- Teleport step size schedule
- Gradient clipping/smoothing strategy near the null cone
- Enumeration and analysis of the 8 (region, degree direction) combinations
- Initial value and bounds for the 5th parameter (minimum and maximum allowed degree)

## Implementation Plan

Build the SQP layer (P + SQ + discrete event mechanism), wire it up to a minimal KV (could be a 2-entry table for XOR's 2 tokens), train with standard CE loss on XOR, log the trajectories. No regularizers (those were iter-5's contribution, not relevant here), no cross-vocab term, no specialization classifier. Just the bare minimum needed to see whether the discrete-degree mechanism does something useful.

### Shorthand Terms:

- P: A polynomial lift operation with one discrete parameter learned via null cone threshold event
- p: The parameter for P
- SQ: A split quaternion operation with 4 continuous learned parameters
- sq: The 4 learned parameters for SQ
- SQP: The combination of SQ and P, with 5 total learned parameters
- sqp: The 5 parameters for the SQP
- KV: A key-value cache
- k: An n-dim vector, the key of a KV lookup
- KSQP: A complete system including a KV, SQP, with continuous parameters following gradient descent, and discrete logic for P.

### Gather data

Implement a subset of the KSQP implementation, flagging any implementation detail not explicitly mentioned in this section. Be wary of any normalization/regularization reflex - these tend to create more problems than they solve. Use XOR as a toy model for proof of concept.

Th sqp vectors will need to be given some initial value. p should initialize to 1. sq should be initialized small random, outside the cone. After a p value change, re-initialize to the same value as the experiment's initial sq init. If transitions are unstable, consider gradient-directed teleport as a refinement, with predictions driven by data collection.

P introduces a fixed pre-processing step and gradient descent on sq proceeds normally.

Choose reasonable starting values for LR. No hysteresis at this stage, just choose a reasonable threshold for both sides of the null cone. Choose one threshold as the increment, one as the decrement threshold. Document these chosen values.

- Initialize SQP
- Use k to lookup sqp
- Compute the SQP gradient, run the training cycle.
- Track the path each vector takes through SQ space.
- When sq has crossed either null cone threshold, perform the associated increment or decrement of p parameter, re-initialize the sq parameters.
- After a null-cone crossing for the sqp associated with key k, write the new (reinitialized) sqp back to the KV cache so that the next lookup of k returns the new values rather than the pre-event ones.
- Log that vector's path up to this point, along with the before&after p parameter.
- Continue training.

Expectation: If our initial guess of inc&dec thresholds are wrong, we should see all null-cone instances end with divergence or all-zero vectors. If it's right, there may still be some divergent cases or all-zero vectors, but it will be a small portion of cases, and will likely be fixed with hysteresis or better initialization/reinitialization protocols.

Data analysis: comb through the logged vector paths and correlate them with destination regions and final p values. Expectation: angle of approach will correlate with final quadrant (Verifiable by plotting trajectory direction vs. arrival region for each crossing event)