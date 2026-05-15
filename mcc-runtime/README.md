# mcc-runtime

Inference-time view of an exported `.mcc` component. Depends on
[`mcc-core`](../mcc-core) but uses only its `value/`, `primitive/`,
`op/`, `graph/`, and `serialization/Importer` packages — no
training, no carving, no exporter on the inference classpath.

## API

```java
Component c = Component.load(Path.of("model.mcc"));
Value out = c.infer(Map.of("x", new MatrixValue(...)));
```

`Component.load` reads `graph.json` + `params.bin`, verifies the
SHA-256 in the manifest matches the blob, reconstructs primitives
via `BuiltinPrimitives.defaults()`, restores parameters, and returns
a runnable graph.

For graphs that use custom primitives, pass a registry:

```java
PrimitiveRegistry registry = BuiltinPrimitives.defaults()
    .register("my-op", cfg -> new MyOp(/* parse cfg */));
Component c = Component.load(modelPath, registry);
```

## Graal native-image

Designed to compile under GraalVM `native-image` without
reflection-config metadata. Everything reachable from `Component.load`
+ `infer` uses only:

- Hand-rolled JSON parsing (`serialization/JsonReader`,
  `serialization/JsonWriter`, `serialization/GraphSchemaCodec`).
- Sealed-interface dispatch via Java's `switch` pattern matching
  (no reflection).
- `java.security.MessageDigest.getInstance("SHA-256")` — covered by
  GraalVM's built-in metadata.

No `Class.forName`, `Method.invoke`, `setAccessible`, `ServiceLoader`,
or dynamic proxies anywhere in the inference path.

## What you get on the classpath

- All `mcc-core` value, primitive, op, graph, and importer classes.
- Carver (`mcc-core/carving/`) and trainer (`mcc-core/training/`)
  are *transitively* available because both modules ship in one
  Maven coordinate, but `Component` itself never touches them.
  A native-image build that doesn't reference them will dead-code
  them out.

## Limitations

- Inference only. Training continuation across reload is not in
  the MVP scope.
- Mandate verification on load is not yet wired (a future runtime
  flag will toggle it).
