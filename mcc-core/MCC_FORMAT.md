# `.mcc` Export Format (schema version 1)

A `component.mcc` directory holds one exported graph:

```
component.mcc/
├── graph.json     (topology + parameter manifest, reflection-free JSON)
└── params.bin     (framed binary parameter blob, body SHA-256 in manifest)
```

## `graph.json` schema

```json
{
  "schemaVersion": 1,
  "nodes": [
    {
      "id": "<unique node id within this graph>",
      "primitive": "<registered primitive name>",
      "config": { ... primitive-specific structural config ... }
    }
  ],
  "edges": [
    { "from": "<src node id>", "to": "<dst node id>", "slot": <int> }
  ],
  "rootInputs": [
    { "node": "<node id>", "slot": <int>, "name": "<runtime input name>", "type": "<ValueType name>" }
  ],
  "terminal": "<terminal node id>",
  "parameters": {
    "sha256": "<hex sha-256 of params.bin BODY (excluding header)>",
    "tensors": [
      {
        "node": "<owning node id>",
        "name": "<local tensor name, e.g. 'W', 'b', 'embeddings'>",
        "shape": [int, int, ...],
        "offset": <byte offset into params.bin body>,
        "length": <byte length>
      }
    ]
  }
}
```

Object key insertion order is preserved. Numbers are serialized as
JSON integers when integral (no decimal point); fractional numbers
use standard JSON decimal notation. NaN and infinity are rejected.

## `params.bin` layout

```
+------------+-------+-------+----------+
|  4 bytes   |  4 B  |  4 B  |   4 B    | header (16 bytes total)
| magic MCC1 |  ver  | count | reserved |
+------------+-------+-------+----------+
| body: each tensor as flat little-endian f64,                |
| in the order listed by parameters.tensors, contiguous.      |
+-------------------------------------------------------------+
```

- **Magic**: bytes `0x4D 0x43 0x43 0x01` (`MCC\x01`).
- **Version**: int32 little-endian. Currently `1`.
- **Tensor count**: int32 little-endian. Matches
  `parameters.tensors.length` in `graph.json`.
- **Reserved**: int32, set to `0`.
- **Body**: contiguous tensors, each `length` bytes
  (= `tensor.data.length × 8`). Offsets in the manifest are
  body-relative (after the 16-byte header).
- **Integrity**: SHA-256 of the body bytes only (excluding header)
  is stored in `parameters.sha256`. The importer rejects mismatched
  blobs.

## Versioning

Any breaking change to either file's layout bumps `schemaVersion`.
The importer rejects unknown versions; renaming a field is a
breaking change.

## Primitive configs

Each built-in primitive declares its own `config` shape. Examples:

| Primitive | `config` fields |
|---|---|
| `add`, `relu`, `softmax`, etc. (stateless) | `{}` |
| `linear`, `linear-no-bias` | `{ "outDim": int, "inDim": int }` |
| `mlp-block` | `{ "sizes": [int, ...] }` |
| `transformer-block` | `{ "seqLen", "dIn", "dModel", "dFf", "dOut" }` |
| `embed` | `{ "dim": int, "symbols": [string, ...] }` |
| `int-to-vector` | `{ "vocabSize": int, "dim": int }` |
| `slice` | `{ "from": int, "to": int }` |
| `vector-to-tensor` | `{ "shape": [int, ...] }` |
| `parameter` | `{ "outputType": "<ValueType name>", "shape": [int, ...] }` |

A primitive that takes structural parameters from its constructor
implements `sibarum.mcc.primitive.Configurable`; its `config()`
method returns the JSON map above. The `BuiltinPrimitives` factory
reads the same map at import time. Custom primitives follow the
same pattern.

## What is *not* in the export

- **Initialization seeds.** A trainable's seed only affects pre-load
  initialization; the saved tensors overwrite that, so seeds aren't
  preserved.
- **Optimizer state.** Only forward parameters are exported;
  gradient buffers and learning rates are training-time only.
- **Mandate verification by default.** Mandates aren't currently
  serialized; the runtime is inference-only. A future schema version
  may add optional mandate carry-over for verifiable inference.
