# Symbolic Term Rewrite Neural Networks
## A Design Document

---

## 1. Summary

This document describes a meta-architecture for *orchestrating* neural networks and other learned components — discovering when and how they should be composed, what intermediate states should pass between them, and what structural constraints the resulting computation should respect. It is not a replacement for MLPs, transformers, or other learned function approximators. The components remain the workhorses; the framework provides the structure within which they cooperate.

Concretely, the architecture separates two graph objects: a slowly-shaped **transformation graph** that encodes which elementary operations and component compositions are reliable in a domain, and a fast, per-computation **computation graph** that is stitched from copies of transformation-graph fragments to perform a specific task. Nodes in these graphs may be MLPs, transformer blocks, CNN layers, hand-coded operations, or other neural networks — the framework treats them as primitives. What it adds is the *structure* in which those primitives operate.

The central design hypothesis: *useful orchestration structure can be discovered by penalty-driven search through an any-to-any prior, with structural plasticity (extension and pruning) replacing continuous weight optimization as the primary learning mechanism for the orchestration layer.* The search is real and expensive; its value comes from what it produces — a restricted transformation graph and stable intermediate-value definitions — that amortize across all subsequent computations.

### 1.1 What the Framework Is For

The framework targets three value propositions, each of which falls out of the same underlying mechanism:

1. **Specification-driven design.** Engineers specify what relationships exist among components, what intermediate values must be produced along the way, and what constraints must hold. The carving process discovers a working composition that satisfies the specification. This shifts architecture design from a craft of hyperparameter tuning toward a discipline of structural specification.

2. **Ante-hoc interpretability.** Intermediate values can be mandated to carry specific symbolic meaning, with the architecture enforcing the mandates structurally rather than via auxiliary losses. The resulting compositions are inspectable by construction, not by post-hoc analysis of trained weights.

3. **Reusable trained priors.** A trained transformation graph is a shareable artifact — accumulated knowledge about which compositions are reliable in a domain, in directly transferable form — from which any number of task-specific computation graphs can be carved. The orchestration knowledge accumulates separately from any specific deployment.

A fourth property follows from these: a continuous dial, the omission pattern of the transformation-graph generator, controls how much orchestration structure is specified up front versus discovered by carving. This parameterizes the trade between generalizability (less specified, more discovered) and convergence speed on specialized tasks (more specified, less searched). It is a property of the framework, not its purpose.

### 1.2 What the Framework Is Not

The framework is not a competitor to MLPs or other learned function approximators. The often-used construction in which transformation-graph nodes are themselves MLPs, with the carving process discovering connectivity between them, is a degenerate special case useful for demonstration purposes — not the principal application. The principal application is composition of *heterogeneous* primitives (different network types, hand-coded operations, symbolic transformations) into pipelines whose orchestration structure is itself learned and inspectable.

Values flowing through the architecture are not restricted to fixed-shape tensors; they may themselves be structured objects (graphs, parse trees, typed symbolic terms). This makes the architecture genuinely compositional — it can carry symbolic structure between learned components without flattening it into vectors.

---

## 2. Core Concepts

### 2.1 The Two Graphs

**Transformation graph.** A directed graph whose nodes are symbols (or, more generally, typed operations) and whose edges represent elementary transformations from one symbol to another. The transformation graph is defined ahead of time by a generator. It is never fully materialized; it is traversed.

The generator may produce a dense any-to-any graph (no prior knowledge), or a structured graph encoding known invariants, type constraints, or algebraic structure (engineered prior knowledge expressed as omissions).

**Computation graph.** A directed acyclic graph constructed at computation time by stitching together copies of transformation-graph fragments. The computation graph is the actual artifact that processes inputs. It is local to a particular traversal context, and the same transformation-graph node may appear in multiple computation-graph contexts with different neighborhoods.

### 2.2 Acyclicity and Direction

Transformation-graph edges are directed. Computation graphs are DAGs and cannot loop. A forward pass is a single sweep through the carved structure. All recurrence is across computations, in the slow updates to the transformation graph and the structural plasticity of computation graphs over time.

### 2.3 Permanence

Each section of computation graph carries a permanence rating based on the frequency of its successful usage. Permanence governs whether a section persists across computations, gets pruned, or seeds new structure.

### 2.4 Values as Graphs

Intermediate values traversing the architecture need not be vectors or matrices. They may themselves be structured objects with internal graph structure — parse trees, proof states, typed symbolic terms, partial solutions. Operations defined on these values are operations on structure, not approximations via embedding.

### 2.5 The System as Nested Graph Structure

The architecture is not a single network but a stack of graph-structured objects, all participating in the same system and interacting through it. Every level is a graph; the levels differ in what their nodes and edges represent and in the timescale on which they change.

- **Transformation graph** — the slowly-shaped substrate. Nodes are symbols or typed operations; edges are elementary transformations.
- **Computation graph** — the fast, per-computation DAG carved from copies of transformation-graph fragments.
- **Values traversing the computation graph** — themselves graphs in the general case (parse trees, proof states, typed symbolic terms, partial solutions). The "data" flowing through the architecture has internal structure that operations respect.
- **Computation-graph nodes** — may themselves contain neural networks (an MLP node, a learned transformation, a sub-computation). A node is not necessarily a primitive scalar operation; it can be a learned function whose own internal structure is parametric.

Talking about these as "multiple overlaid networks" is shorthand for the fact that the architecture is genuinely nested: graphs of graphs whose nodes may be graphs and whose contained computations may themselves be networks. The action principle operates uniformly across the levels; what differs is the timescale of change and the kind of object being traversed or shaped.

This nesting is what makes the architecture genuinely compositional. A standard neural network has a single representational level — activations on fixed-shape tensors. This architecture has structure at every level, and the structure at one level is the substrate at another.

---

## 3. The Action Principle: What It Is and What It Costs

The action principle is a search procedure, not a magic convergence mechanism. It is honestly expensive, and the document should not pretend otherwise. The framework's value does not come from the search being cheap; it comes from what the search produces — a restricted transformation graph and a set of stable intermediate-value definitions that pay back the search cost across all subsequent computations.

### 3.1 Penalty and Search

Each transformation-graph edge accumulates penalty in two main ways:

- **Attempts that never materialize.** Edges proposed during carving that fail to lead anywhere useful — the local action looked acceptable but the path did not extend.
- **Attempts that lead to dead-ends.** Edges that participated in computations terminating without reaching a target, or that contributed to paths penalized under failure.

These penalties cause the system to attempt more and more potential routes. The search is real and expensive: many routes are tried before one is found that is more reliable than the others. The mechanism is not "find the right path quickly" — it is "attempt many paths, accumulate evidence about which fail, and let the failures shape what gets attempted next."

Less reliable routes are not pruned eagerly. They are kept alive as alternatives until a more reliable route is found, at which point the less reliable routes are pruned in comparison. Pruning is *competitive*, not absolute. A route is bad relative to a better route through the same region, not bad on its own.

### 3.2 Distance, in Two Forms

Distance is what makes penalty actionable — without a notion of progress toward a target, penalty is just a record of failures with no direction for improvement. The framework uses distance in two complementary forms:

**Graph-traversal distance.** The directed distance from a node to a target along the transformation graph itself. This is universal — defined for any pair of nodes that are connected — but it is *expensive*: computing it on a partially-materialized graph requires search, and search is the cost the framework pays for structural discovery.

**Intermediate-value proximity.** When intermediate states have been defined, distance can also be measured by how close a current value is to a mandated intermediate or final target *in value space*. This is much cheaper than graph traversal, because it operates on the values themselves and does not require knowing the graph route between them.

The two distance forms are complementary. Graph-traversal distance is what the system uses when no useful proximity metric is available — early in training, in regions of the graph where intermediate states have not yet stabilized, or for symbol pairs whose value-space relationship is unclear. Intermediate-value proximity is what the system uses once intermediate states are defined: it provides a dense, cheap signal that bypasses graph search entirely.

This is one of the major payoffs of defining intermediate states. Before they exist, distance is expensive; after they exist, distance becomes a cheap operation on values, and the search burden of the action principle drops accordingly.

### 3.3 The Real Payoff

The action principle's enormous cost — running a real search through accumulating penalties until reliable routes emerge — is justified by what the search produces, not by the search being efficient. The search yields two structural assets:

- **A restricted transformation graph.** Penalty accumulation prunes the dense any-to-any prior down to the subset of edges that have proven reliable for the domain. The restricted graph is much smaller than the prior, much faster to traverse, and reusable across all future tasks in the domain.
- **Defined intermediate states.** Once the search has identified stable intermediate values that recur across successful traversals, those intermediates can be promoted to first-class structural elements of the architecture. They become anchors for future carvings, cheap distance landmarks, and inspectable interpretability points.

These two outputs amortize the search cost. The transformation graph is trained once (or extended incrementally over time) and reused across many computation-graph carvings. Intermediate states defined during training accelerate every subsequent traversal that uses them. The framework spends compute up front to produce structure that pays back over the artifact's lifetime.

The action principle is therefore best understood as a *structural learning algorithm* — analogous to discovering useful abstractions or named subroutines in symbolic systems — rather than as a forward-pass optimization. Its expense is real and is the price paid for structure that subsequent operations can exploit.

### 3.4 Two Levels of Application

The action principle operates at both graph levels, on different timescales:

- On the **transformation graph**, accumulated penalty governs which edges remain available for future carving. This is slow, structural, and reusable across tasks.
- On the **computation graph**, the same evaluation governs which candidates get instantiated and stitched for a particular computation. This is fast and exploratory, local to a single traversal.

The transformation graph is the slow asset; each computation is a trajectory across it; the trajectory contributes evidence that updates the asset.

---

## 4. Learning Dynamics

### 4.1 Path Tracking

As a symbol or network traverses a computation graph, it tracks the path taken. At termination, the final form is compared to the expected value, with closeness measured by either form of distance from Section 3.2 — value proximity when intermediate states have been defined, graph traversal otherwise:

- **Success (final form matches expected value):** the section of graph used is reinforced.
- **Partial success (final form is closer to expected value than the starting symbol):** the section is reinforced proportionally to the progress made.
- **Failure (final form is further from expected value than the starting symbol):** the path is penalized.

This produces a *dense* learning signal where most rewrite systems have only a sparse terminal one. Even traversals that fail to reach the target contribute usable evidence, provided some notion of progress can be measured.

### 4.2 Penalty as Plasticity

Penalty is not synonymous with pruning. A penalized section becomes a candidate for structural change, the nature of which depends on the local success-versus-failure rate. The same penalty signal can drive:

- **Pruning** of consistently unreliable sections.
- **Reinforcement decay** of sections that are mixed.
- **Expansion** — creation of new transformation-graph structure — where penalty pressure is high but traversal continues to attempt the region (signaling a structural deficiency rather than a bad path).

### 4.3 Two Credit-Assignment Regimes

Credit assignment operates differently on each graph:

- **Transformation-graph edges carry a prior** over which elementary moves are worth materializing. Penalties here are aggregated across many computation-graph instantiations.
- **Computation-graph paths carry a posterior** for a specific composition in a specific context.

A pathway can be bad even when its constituent edges are individually good (composition fails); a pathway can succeed despite containing locally-disfavored edges (context redeems them). Keeping these regimes separate lets the architecture express both. The aggregation rule that connects computation-graph evidence to transformation-graph penalties is itself a design choice, ranging from conservative (penalize the underlying edge only with broad evidence across contexts) to aggressive (any consistent failure pattern propagates).

### 4.4 Unreachability and Generative Failure

When traversal terminates without a path to the expected value, distance is not computable. From the last node with reachability to the target, one of two things happens:

- A new branch is created — extending the transformation graph in the direction of the target.
- If available branches all carry penalty high enough that the cost of pursuing them exceeds the computation budget, the computation graph is pruned.

The decision is governed by the same penalty-and-distance evaluation that governs traversal everywhere else. Unreachability is generative rather than degenerate: it triggers either targeted graph extension or principled contraction.

### 4.5 Self-Extension Under the Action Principle

When the transformation graph extends itself, it does so by the same minimization that governs traversal. New edges are proposed by the generator, in concert with training data when available, weighted by least distance × penalty in the direction of the target. There is no separate generator-of-generators; the same mechanism handles traversal and growth, and the mechanism is the same minimization in both cases.

This gives the architecture a self-bootstrapping property: it builds the graph it needs to evaluate whether it built the right graph. The transformation graph is both the substrate and the yardstick, and they co-evolve under a single principle.

---

## 5. Bias by Omission

### 5.1 The Inversion

Inductive bias is conventionally injected by *adding* structure: convolutional connectivity, equivariance constraints, attention masks, regularizers, typed embeddings. Each addition is a positive claim about how the structure should behave.

This architecture inverts the convention. Prior knowledge is encoded by which edges the transformation-graph generator *declines to produce*. Everything that remains is fully available to the carving process, with no tilt toward any particular use.

### 5.2 Properties of Omission-Based Priors

- **Epistemic humility.** "This transformation isn't useful" is a weaker claim than "this transformation should behave like X." Wrong omissions cost expressiveness but do not introduce active interference.
- **Compositionality.** Multiple omissions intersect cleanly. Three independent constraints (type incompatibility, grade preservation, invariant preservation) compose to a smaller candidate space without fighting each other.
- **Inspectability.** The bias can be read off the generator directly. Asking "what does this prior rule out?" has a definite answer: exactly these edges are not in the graph.
- **Granularity.** Omission is expressible at many levels — specific edges, edge types, whole subgraphs, certain compositions, certain reachability relations, certain grades — using the same formalism.

### 5.3 A Calculus of Priors

Two researchers' priors can be combined by union (more permissive) or intersection (more restrictive). Priors can be ablated by un-omitting specific edges and observing whether the carved graph changes. The strength of a prior is quantifiable as the count of omitted edges. None of this is available with additive biases in any clean form.

### 5.4 The Risk to Note

Omission can be wrong in a way that is harder to detect than additive bias. Removing a load-bearing edge produces silent absence rather than fighting gradients. Validation strategies — periodic sampling of un-omitted edges, holding out a small set of omissions, monitoring whether the system attempts paths that would benefit from removed edges — are part of the engineering practice the framework requires.

---

## 6. Mandated Intermediate Values

### 6.1 Mandates Are Non-Local

A mandate specifies that a computation must produce a particular value, with particular semantic meaning, somewhere in the carved graph. Mandates are *not* pinned to a specific computation-graph location. They specify *what* must be produced, not *where*.

This is an important architectural choice. Pinning a mandate to a fixed location would over-constrain the carving — the system would be forced to produce the mandated value at exactly that point, even when an equivalent value could be produced earlier or later by a more efficient path. Allowing mandates to occur anywhere in the carved graph means the carving discovers the appropriate location as part of its structural search.

A mandate is therefore better understood as an **initial target** — a value the system must produce in the course of solving the task, with the location of its production determined by the same action-driven search that shapes everything else about the carved graph. Multiple mandates create multiple initial targets; the carving finds an arrangement that satisfies all of them.

### 6.2 Initial Targets and Search Decomposition

Treating mandates as initial targets has a significant consequence: it decomposes a hard search problem into a sequence of easier ones.

Without mandates, the carving must find a path from input to final result through a large unstructured space. With mandates, the carving has multiple intermediate targets to reach, each of which is closer to the input than the final result is. The search becomes "find a path to the first mandated value, then from there to the next, then to the result" — a chain of shorter searches rather than one long one.

This is one of the principal reasons the framework is workable in practice despite the action principle's expense. Mandates do not just provide interpretability; they provide search structure. They turn an end-to-end discovery problem into a sequence of localized discovery problems, each with its own target and its own opportunity for the carving to succeed.

### 6.3 Mandates Define Cheap Distance

Once intermediate states are defined and stable, they provide the second form of distance described in Section 3.2: proximity in value space. Distance from "current value" to "nearest unsatisfied mandate" can be computed directly on the values, without graph traversal.

This compounds the search-decomposition benefit. Not only does the carving have shorter searches to perform, but those searches are guided by a cheap proximity metric rather than by expensive graph-traversal distance. The shift from traversal-distance to value-distance is a phase change in the cost of the action principle's search, and it is what mandates buy in addition to interpretability.

### 6.4 Structural Enforcement, Not Soft Constraint

Mandates are enforced structurally, not through auxiliary loss terms. A carved graph either produces the mandated value somewhere along its execution, or it does not satisfy the mandate and is rejected by the carving process. There is no soft trade-off where the system can choose to violate a mandate to reduce some other loss; mandates are hard.

The mechanism: paths that do not produce a mandated value at any point are not eligible for stitching. The carving process treats mandated values as required waypoints — required to occur, but not required at a fixed location. This is structural enforcement of a non-local constraint, which is qualitatively different from both fixed-architecture interpretability (which fixes locations but not values) and loss-based constraint enforcement (which is soft and tradeable).

### 6.5 Specification-Driven Training

A training example in this framework is richer than an input-output pair. It is:

- **Payload** — input, which may itself have rich structure.
- **Mandated intermediates** — named, typed values the computation must produce somewhere along its execution.
- **Result** — the desired output (itself a mandate, distinguished only by being the final target).
- **Constraints** — symbolic and numerical restrictions on what counts as a valid computation (types, ranges, invariants, conservation laws).

The carving operates within the joint specification, finding a graph that produces the result, produces every mandated intermediate somewhere, and satisfies every constraint. The resulting computation is verifiable against its specification rather than only evaluated against held-out data.

### 6.6 Diagnostic Failure

Failure modes split cleanly. If carving fails, the architecture can localize the failure: this mandated value is not produced by any reachable path from the inputs, given the available transformations. The diagnostic is precise even though the mandate's location is non-local — the question is "can this value be produced anywhere in a valid path?", which has a definite answer from the carving's search history. That tells the user whether the symbolic skeleton is wrong, the data is inadequate, or the transformation graph is missing edges. Standard architectures collapse all such failure modes into "the loss did not converge."

### 6.7 Curriculum Through Supervision Relaxation

Strong-mandate training (many intermediates specified) gives fast convergence on a small carved graph because each mandate decomposes the search and provides a cheap distance signal. Progressively relaxing mandates lets the carving rediscover those intermediates on its own and may reveal alternative paths the original mandates did not anticipate. This is curriculum learning at the level of *how much structure is specified*, and it is also a natural way to amortize the action principle's search cost — start with strong supervision to get the structure approximately right, then relax to let the system find improvements.

---

## 7. Reusable Trained Transformation Graphs

### 7.1 Decoupled Artifacts

A trained MLP fuses learned structure with deployable network. A trained transformation graph does not. It is a *prior over which edges are reliable*, accumulated across however many computation graphs were carved from it. The deployable artifacts (computation graphs) are downstream; any number can be produced from the same underlying transformation graph.

### 7.2 Transferable Domain Knowledge

A transformation graph trained on varied tasks within a domain encodes which transformations are reliable in that domain, independent of any specific task. A new task within the same domain carves its computation graph from this trained prior, gaining the benefit of accumulated reliability evidence without rediscovering it.

This is structurally similar to pretraining-and-finetuning, but operationally cleaner. Catastrophic forgetting does not occur, because the new task does not overwrite the transformation graph; new evidence accumulates monotonically into the same prior.

### 7.3 Scaling

Capability accumulation and specific deployments operate on different timescales. Scaling up means training the transformation graph longer and across more tasks, accumulating an increasingly refined prior. Specific deployments carve specialized computation graphs from this rich prior on demand. The transformation graph is the long-lived asset; computation graphs are short-lived artifacts.

### 7.4 Knowledge Sharing

Trained transformation graphs are shareable in a way trained MLPs are not. A recipient is not bound to architectural choices of the producer; they receive accumulated *evidence about the domain* and may carve their own computation graphs with their own mandated intermediates, omissions, and constraints. The merging of multiple trained transformation graphs (across researchers, datasets, tasks) is cleaner than model-merging because the underlying object is just per-edge reliability statistics.

### 7.5 The Stability Question

Reusability requires that the transformation graph's edge structure is stable across the tasks being aggregated. This is an empirical and design question — finding the granularity at which a "domain" is coherent enough to support cross-task evidence accumulation. Too narrow and reusability benefits do not accrue; too broad and accumulated evidence is incoherent.

---

## 8. The Framework as an Architecture-Composition Tool

The framework's primary purpose is composing heterogeneous components — different network architectures, hand-coded operations, symbolic transformations — into pipelines whose orchestration structure is itself learned and inspectable. This section describes how various component types fit into the framework, with the self-assembling case treated as a useful but degenerate illustration.

### 8.1 Components as Transformation-Graph Nodes

Any computation with well-defined input and output types can be a node in the transformation graph. The framework does not care what is inside the node; it cares about what the node consumes, what it produces, and how reliably it contributes to successful traversals. This allows mixed compositions:

- An **MLP node** does numerical function approximation on tensor data.
- A **transformer block** processes sequence-structured input.
- A **CNN layer** processes spatially-structured input.
- A **GNN module** processes graph-structured input.
- A **hand-coded operation** (parser, tokenizer, type converter, lookup) handles deterministic transformations the engineer prefers not to learn.
- A **symbolic transformation** (rewrite rule, simplification, substitution) handles structured manipulations on graph-valued data.

The carving process operates over compositions of these primitives. It does not need to know that one node is an MLP and another is a parse-tree rewrite; it needs to know their type signatures and the penalty evidence accumulating on edges between them. Heterogeneity at the primitive level is handled uniformly at the orchestration level.

### 8.2 What the Carving Adds

Given a library of primitives and an any-to-any prior over their compositions, the carving process discovers:

- **Sequencing.** Which primitive should follow which, given the type-correctness constraints and the accumulated evidence about which compositions reliably make progress toward targets.
- **Branching and merging.** Where the computation should split across alternative paths and where parallel paths should converge.
- **Skip connections and shortcuts.** Direct compositions that bypass intermediate primitives when they are reliable enough to be worth keeping.
- **Width and capacity allocation.** Where the carved structure should elaborate (more primitives applied, more alternative paths kept) versus contract (single canonical path).

These are decisions an engineer would otherwise make by hand, by hyperparameter search, or by neural architecture search. The framework makes them via the same penalty-and-distance mechanism that handles everything else.

### 8.3 Mandates as Component Interfaces

Mandated intermediate values function as typed interfaces between components. When a mandate specifies that a particular structured value must appear somewhere in the computation, it is implicitly declaring that some upstream component must produce it and some downstream component must consume it. The carving discovers which components fill those roles; the mandate ensures the interface is respected.

This is the architectural feature that makes heterogeneous composition genuinely workable. Without mandates, composing an MLP with a symbolic rewrite system is hard because their representational conventions differ. With mandates, the interface between them is a named, typed value with specific semantic meaning — and either the components produce/consume that interface correctly or the carving fails to find a valid composition. The interface is a hard structural constraint rather than a learned alignment.

### 8.4 The Specialization Dial

The omission pattern of the transformation-graph generator controls how much orchestration structure is specified up front:

- **Few omissions (dense any-to-any over the primitive library):** the carving has maximum freedom to discover compositions; the search is correspondingly more expensive. Useful for exploratory work or when the engineer has no strong priors about which compositions should be allowed.
- **Many omissions (highly restricted prior):** most compositions are ruled out by construction; the carving has a small space to search and converges quickly. Useful for deployment where the engineer knows which compositions make sense and wants the carving to handle only the residual structural decisions.
- **Intermediate:** the typical engineering case. Type-correctness and known invariants are encoded as omissions; the carving discovers the rest.

This is a property of the framework, not a competitive feature. It means engineers can dial how much they specify versus how much they delegate, and they can do so smoothly and inspectably. The dial is not a knob that trades performance against generality; it is a knob that trades engineering effort against search cost.

### 8.5 The Self-Assembling Single-Architecture Case

A degenerate but illustrative configuration: the primitive library contains only one type of component, and the carving discovers how to compose copies of it.

Take the case where every node is a matmul+ReLU. The carving discovers depth, skip-connection topology, and width-allocation across depth. ResNets emerge as carvings that favor edges bypassing intermediate nodes; depth becomes emergent rather than chosen; width allocation becomes adaptive.

This case is useful as a clean experimental vehicle — same primitives as standard MLP, only the connectivity is carved — but it is not the principal application. The interesting use of the framework is heterogeneous composition, where the primitives differ and the carving's job is orchestration across that heterogeneity. Single-primitive carving is a sanity check that the mechanism works; multi-primitive carving is what the mechanism is for.

---

## 9. Demonstration: Arithmetic Expression Evaluation

### 9.1 Why This Task

Arithmetic expression evaluation (e.g., `5283+(2134*17)`) is an effective demonstration of the framework's principal application: orchestrating heterogeneous primitives into a working pipeline whose structure is discovered rather than designed.

The task naturally requires several different *kinds* of computation — string manipulation, numerical parsing, type conversion, learned arithmetic, structured value handling — none of which is best served by a single architecture. The demonstration shows the framework discovering how to compose these heterogeneous primitives into a coherent pipeline, with verifiable intermediate states, from a typed any-to-any prior.

The task is also:

- Constrained enough that correct answers are verifiable and intermediate states are nameable.
- Rich enough to require real structural decisions (precedence, nesting, recursion).
- Aligned with how the framework is intended to be used in practice — as a tool for orchestrating mixed components, not for replacing any single one.

### 9.2 Specification

- **Payload:** the input string.
- **Node types (the primitive library):** `split-string-at-character`, `parse-number`, `number-to-matrix`, `MLP`, `compose-matrices`, `matrix-to-number`, `output`. Each represents a different kind of computation; the MLP nodes are the only learned components, and they handle the actual arithmetic.
- **Connectivity prior:** any-to-any over node types, modulo type compatibility (see 9.3).
- **Mandated intermediates:** the token list after splitting; the parsed numerical values; the result of inner-paren evaluation before outer operations; intermediate operation results respecting precedence.
- **Result:** the numerical value of the expression.
- **Constraints:** type signatures of each node type; precedence rules expressible as ordering constraints on mandates.

### 9.3 Type Structure

Each node type has typed inputs and outputs. The any-to-any prior is over node-type compositions, not over arbitrary input-output connections — a `compose-matrices` node cannot accept a string. Edge omission encodes type compatibility from the start, aligned with the bias-by-omission principle.

### 9.4 What the Demonstration Shows

The headline claim:

- **The framework discovers a working pipeline that orchestrates MLP primitives with hand-coded operations and structured-value transformations, with verifiable intermediate states, from a typed any-to-any prior.**

Subsidiary claims tested by the same demo:

- The carving process discovers correct sequencing (string-handling before parsing, parsing before arithmetic, arithmetic before output formatting) from penalty evidence alone.
- Mandated intermediate values are produced correctly by the carved graph, providing direct evidence for the ante-hoc interpretability claim.
- Values that are themselves graphs (parse trees) flow through the carved structure as first-class objects, demonstrating the values-as-graphs property in a concrete setting.
- The MLP nodes in the carved graph do exactly what MLPs are good at — learning the actual arithmetic — while the framework handles the orchestration that surrounds them.

### 9.5 Progression

The demonstration scales smoothly:

1. **Addition only** — minimal pipeline. Establishes that the basic orchestration works.
2. **Multiplication added** — operator precedence emerges in the carved graph (or is mandated, and the carving respects the mandate).
3. **Parentheses** — nested evaluation forces the carved structure to handle recursion, either learned structurally or via repeated string-splitting of sub-expressions.

Each step is a separate experimental claim with a clear pass/fail criterion. Each step also adds heterogeneity — more primitive types, more interface points between them — which exercises the framework's principal capability rather than incidental ones.

### 9.6 Ablations

- **Replace MLP nodes with hand-coded arithmetic.** If carving still works, the carving is doing orchestration (discovering structure around primitives), not learning arithmetic disguised as structure.
- **Hand-code the pipeline; let only the MLP weights train.** Establishes a baseline: how much of the result is the framework discovering structure versus how much is the MLPs learning the underlying function. The framework's value is the difference.
- **Vary the omission count of the transformation-graph generator.** Produces an empirical curve showing how engineering effort (more omissions) trades against search cost (less omissions, more carving work).
- **Swap one primitive for another of the same type signature.** Replace MLP nodes with small transformer blocks, or with a different learned component. If the carved structure transfers (the orchestration is the same, only the underlying learner changed), this directly demonstrates the framework's component-agnostic nature.

The fourth ablation is the most informative one for the framework's positioning. It shows that the orchestration discovered by carving is genuinely about *structure* and *interfaces*, not about which specific learned component happens to be inside each node. That is the strongest demonstration of the framework's intended use.

---

## 10. Engineering Concerns and Open Questions

### 10.1 Stability of the Action Principle

The action principle's stability under rich joint supervision (output, intermediates, and constraints simultaneously) is the central empirical risk. The space of valid carved graphs may be small enough that the per-edge least-action signal is hard to follow; local minima where most-but-not-all mandates are satisfied may be hard to escape. Annealing schedules on mandates, structural priors that bias toward satisfiable topologies, and rollback mechanisms for stuck carvings are likely additional ingredients.

### 10.2 Aggregation Rule for Cross-Graph Credit

How computation-graph penalties propagate to transformation-graph edges shapes system behavior significantly. Conservative aggregation produces slow but stable adaptation; aggressive aggregation produces faster but more thrashing structural change. Whether this rule should be fixed, scheduled, or itself learned is open.

### 10.3 Heuristic Distance Estimation

Computing exact distances on a partially-materialized transformation graph is expensive at scale. A learned heuristic — a value function approximating distance from any node to any target, refined by actual traversal outcomes — is likely necessary in practice. This is a natural role for an overlaid network.

### 10.4 Branch-Creation Policy

When unreachability triggers expansion, the proposal mechanism for new edges is governed by the transformation graph itself, weighted by least distance × penalty in the direction of the target. The meta-signal — whether expansion attempts at a given location historically succeed or fail — should feed back into the expand-versus-prune decision, producing a localized prior on graph completability. The exact form of this meta-update is a design choice.

### 10.5 Stability of Trained Priors

The reusability story depends on trained transformation graphs encoding domain-general rather than task-specific structure. Whether penalties capture transferable knowledge or task-specific noise is the make-or-break question for the third value proposition. The test is conceptually simple: train a transformation graph on task A, carve a computation graph for task B, compare against carving from scratch.

### 10.6 Infrastructure Cost

Implementing the carving mechanism — action minimization, penalty accumulation, graph extension and pruning, stitching mechanics — is substantial engineering. The risk of spending months on infrastructure before getting empirical signal is real. A minimal version on the simplest task (addition only, no parens) within a short timeframe is the right first milestone; if even that requires extensive infrastructure investment, the framework's complexity is higher than it appears, which is itself information.

---

## 11. Positioning

The framework occupies a position in the design space that current tools do not serve well: it is a *meta-architecture* — a tool for orchestrating other neural networks and computational primitives — rather than a network architecture in its own right.

What this framework offers, that the standard toolkit does not:

- **Heterogeneous composition.** Networks of different types, hand-coded operations, and symbolic transformations can be composed into pipelines whose orchestration structure is itself learned. Mandated intermediate values serve as typed interfaces that make the composition workable.
- **Specification-driven design.** Engineers specify what relationships exist, what intermediate values must be produced, and what constraints must hold. The carving discovers a working composition that satisfies the specification. Architecture choice becomes structural specification rather than craft.
- **Verifiable interpretability.** Mandated intermediate values are produced by structural enforcement, not approximated by training pressure. The carved structure is inspectable by construction.
- **Reusable orchestration knowledge.** A trained transformation graph encodes accumulated evidence about which compositions are reliable in a domain — separately from any specific deployment, and shareable across teams without architectural lock-in.

The framework is complementary to the existing toolkit, not competitive with it. MLPs, transformers, CNNs, GNNs, and other learned components remain the workhorses; the framework provides the structural layer in which they cooperate. The relationship between engineer and model shifts: instead of choosing a single architecture and tuning hyperparameters, the engineer specifies the primitive library, the structural priors (as omissions), and the required intermediate states, then lets the carving discover the orchestration.

Whether this shift is empirically justified across enough domains to matter is the work ahead. The framework's value depends on whether the discovered orchestrations are good — meaning both correct (they satisfy mandates and produce right outputs) and worth the search cost (the resulting structure pays back its discovery expense across enough subsequent computations to justify the up-front investment). Both questions have to be answered empirically, on real tasks, with real primitive libraries. The arithmetic demonstration in Section 9 is the first such test; broader applications, particularly those with richer primitive libraries and structured intermediate values, are where the framework will either prove its place or reveal its limitations.

---

## 12. Glossary

- **Transformation graph** — predefined directed graph of elementary operations; the substrate from which computation graphs are carved.
- **Computation graph** — DAG of stitched transformation-graph fragments; the artifact that processes inputs.
- **Permanence** — frequency-of-successful-usage rating that governs persistence of computation-graph sections.
- **Action principle** — the search procedure that uses accumulated penalty and available distance metrics to guide traversal, carving, and structural extension. Expensive in itself; justified by the structural assets it produces.
- **Distance** — measure of progress toward a target. Two complementary forms: graph-traversal distance (universal but expensive) and intermediate-value proximity (cheap, available once intermediate states are defined).
- **Penalty** — accumulated evidence of unreliability on transformation-graph edges and computation-graph paths. Drives competitive pruning and search expansion.
- **Mandate** — required intermediate value the computation must produce somewhere along a valid path. Non-local: specifies *what*, not *where*. Functions as an initial target that decomposes the search and provides cheap distance landmarks.
- **Omission** — an edge or class of edges the transformation-graph generator declines to produce; the unit of bias.
- **Carving** — the process by which evidence shapes the materialized portion of the transformation graph and the structure of computation graphs.
- **Stitching** — the construction of a computation graph by joining copies of transformation-graph fragments, respecting directed-edge type compatibility.
