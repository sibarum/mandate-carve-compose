# mcc-core

Reflection-free, Graal-native-image-friendly library for designing,
training, and exporting small experimental NN components. Consumed by
the eventual Mandate-Carve-Compose GUI; usable today as a plain Java
library.

- **Java 25.** No external runtime dependencies. Hand-rolled JSON,
  `double[]` tensors, from-scratch backprop.
- **Forward + backward + carving** all live here.
  [`mcc-runtime`](../mcc-runtime) is the inference-only counterpart.

## Value types

Every primitive consumes and produces values from a sealed
`Value` hierarchy (`sibarum.mcc.value`):

| Type | Java record | Shape |
|---|---|---|
| `STRING` | `StringValue(String s)` | n/a |
| `NUMBER` | `NumberValue(double n)` | scalar |
| `MATRIX` | `MatrixValue(double[] data)` | 1-D vector |
| `TERNION` | `TernionValue(x, y, z)` | 3 components |
| `QUATERNION` | `QuaternionValue(a, b, c, d)` | 4 components |
| `TENSOR` | `TensorValue(int[] shape, double[] data)` | n-D row-major |

`MatrixValue` is the workhorse 1-D type carried over from
strnn-model. New shaped data uses `TensorValue`. Reshape primitives
(`vector-to-tensor`, `tensor-to-vector`, `vector-to-ternion`, …)
bridge between them.

## Primitive contracts

Three layered interfaces in `sibarum.mcc.primitive`:

- **`Primitive`** — `name()`, `inputTypes()`, `outputType()`,
  `apply(List<Value>)`. Optional `inversion()` for the carver.
- **`Differentiable extends Primitive`** — adds
  `List<Value> backward(Value gradOutput)`. Returns input-slot
  gradients; `null` for slots whose input type has no continuous
  gradient (e.g. discrete indices feeding `IntToVector`).
- **`Trainable extends Differentiable`** — adds `step(double lr)` and
  `trainableIdentity()`. The same `backward` call both produces
  input gradients and accumulates parameter gradients into the
  trainable's internal buffers; `step` applies them.

Two orthogonal contracts for serialization:

- **`Parameterized`** — declares the trainable's parameter tensors
  for export/import.
- **`Configurable`** — declares structural parameters (sizes, shapes,
  slice ranges) that go into the JSON config.

## Built-in primitives

`BuiltinPrimitives.defaults()` returns a registry populated with
everything below. The GUI's export → reload path uses this; a custom
GUI extension can `.register(...)` additional primitives on top.

**Stateless arithmetic + activations** (Differentiable, no parameters):

`Add`, `Sub`, `Mul`, `DotProduct`, `Magnitude`, `Normalize`,
`Softmax`, `Sigmoid`, `Tanh`, `Relu`, `CosineSimilarity`,
`SimilarityGate`, `Concat`, `Slice` (configurable range),
`CrossProduct3` (on `TernionValue`), `MatMul`, `MatVecMul`.

**Reshape** (pure shape conversion):

`VectorToTernion`, `TernionToVector`, `VectorToQuaternion`,
`QuaternionToVector`, `VectorToTensor` (configurable shape),
`TensorToVector`.

**Data mappings**:

`Embed` (string → vector, trainable), `Lookup` (vector → nearest
string), `IntToVector` (integer index → vector, trainable),
`VectorToInt` (argmax).

**Trainable primitives**:

`Linear` (affine map, optional bias), `Parameter` (no-input
output-only learnable value of any Value type), `MlpBlock`,
`TransformerBlock`.

**Algebra utilities** (called by primitives, not directly graph nodes):

`op/advanced/Mat2` (2×2 helpers), `op/advanced/SplitQuat` (4D
split-quat with full backward), `op/advanced/PolyLift` (Π-net
monomial lift with backward).

**Carver scaffolding**:

`Terminal` — identity primitive used as the carved graph's terminal.

## Building a graph

```java
// 1. Build the substrate — a palette of named primitives.
TransformationGraph tg = new TransformationGraphBuilder()
    .addNode("mlp",  new MlpBlock(new int[] { 2, 16, 16, 2 }, 7L))
    .build();

// 2. Build an execution graph.
CompGraphNode node = new CompGraphNode("c-mlp", tg.node("mlp"));
ComputationGraph cg = new ComputationGraph(List.of(node), node);

// 3. Bind roots, execute.
cg.bindRoot(node, 0, new MatrixValue(new double[] { 0.3, -0.7 }));
Value out = cg.execute();
```

## Training

```java
GraphTrainer trainer = new GraphTrainer(
    cg,
    (g, ex) -> g.bindRoot(node, 0, (MatrixValue) ex.inputs().get("x")),
    /* learningRate */ 0.05,
    /* stepEveryExample */ true
);
double avgLoss = trainer.trainEpoch(corpus);
```

`GraphTrainer` does proper reverse-topological gradient flow.
Multi-trainable chains (Linear → Relu → Linear → Softmax) all train
in one backward pass; trainables sharing the same `trainableIdentity`
have `step` called once per example.

## Mandates

A `MandateSet` declares values the executed graph must produce. The
result mandate is matched against the terminal node; intermediate
mandates can match any node in the DAG; ordering constraints are
checked via topological reachability.

```java
MandateSet mandates = new MandateSet(List.of(
    Mandate.result(targetVector, 0.01, /* ordering */ 10),
    Mandate.intermediate("preActivation", expectedHidden, 0.1, /* ordering */ 5)
));
VerificationReport report = new MandateVerifier().verify(cg, mandates);
```

## Carving

`BackwardChainingCarver` builds a `ComputationGraph` from the
substrate that *could* produce the mandate target. Per-primitive
inversion lives on the primitive itself via `Primitive#inversion()`;
the carver contains no primitive-specific branches.

```java
BackwardChainingCarver carver = new BackwardChainingCarver(seed);
CarvingResult result = carver.carve(tg, mandates, rootInput);
ComputationGraph carved = result.graph();
```

The returned `CarvingResult.simulatedValues()` map gives each node's
expected output value — useful as a training target.

## Export & reload

```java
new Exporter().export(cg, List.of(
    new Exporter.RootInput("x", node, 0)
), Path.of("model.mcc"));

// In mcc-runtime:
Component c = Component.load(Path.of("model.mcc"));
Value out = c.infer(Map.of("x", new MatrixValue(...)));
```

The export format is documented in [`MCC_FORMAT.md`](MCC_FORMAT.md).

## Adding a custom primitive

```java
public final class MyOp implements Differentiable, Configurable {
    @Override public String name() { return "my-op"; }
    @Override public List<ValueType> inputTypes() { return List.of(ValueType.MATRIX); }
    @Override public ValueType outputType() { return ValueType.MATRIX; }
    @Override public Value apply(List<Value> inputs) { /* forward */ }
    @Override public List<Value> backward(Value gradOutput) { /* input grads */ }
    @Override public Map<String, Object> config() { return Map.of("foo", 42); }
}

PrimitiveRegistry registry = BuiltinPrimitives.defaults()
    .register("my-op", cfg -> new MyOp(/* parse cfg */));
Component c = Component.load(modelPath, registry);
```

Add `Trainable` if your op holds learnable parameters; add
`Parameterized` if those parameters need to round-trip through export.

## Runnable examples

- `examples/SineRegressionExample` — build → train → export → reload →
  infer for a 2-layer MLP.
- `examples/CarveThenTrainExample` — declare a mandate, carve a graph
  from the substrate, train it to satisfy the mandate.

Run with:

```
mvn exec:java -pl mcc-core -Dexec.mainClass=sibarum.mcc.examples.SineRegressionExample
```

## Tests

```
mvn test -pl mcc-core,mcc-runtime
```

Includes gradient checks at ≤ 1e-7 max-abs-err for every Differentiable,
sealed-permits round-trips for value types, end-to-end carve → train
+ export → reload + inference parity tests.
