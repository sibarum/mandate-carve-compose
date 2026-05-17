package sibarum.mcc.primitive;

import sibarum.mcc.op.Add;
import sibarum.mcc.op.Concat;
import sibarum.mcc.op.CosineSimilarity;
import sibarum.mcc.op.CrossProduct3;
import sibarum.mcc.op.DotProduct;
import sibarum.mcc.op.HarmonicLift;
import sibarum.mcc.op.IntToVector;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.Magnitude;
import sibarum.mcc.op.MatMul;
import sibarum.mcc.op.MatVecMul;
import sibarum.mcc.op.Mul;
import sibarum.mcc.op.Normalize;
import sibarum.mcc.op.Parameter;
import sibarum.mcc.op.Relu;
import sibarum.mcc.op.Sigmoid;
import sibarum.mcc.op.SimilarityGate;
import sibarum.mcc.op.Slice;
import sibarum.mcc.op.Softmax;
import sibarum.mcc.op.Sub;
import sibarum.mcc.op.Tanh;
import sibarum.mcc.op.TensorToVector;
import sibarum.mcc.op.TernionToVector;
import sibarum.mcc.op.QuaternionToVector;
import sibarum.mcc.op.VectorToInt;
import sibarum.mcc.op.VectorToQuaternion;
import sibarum.mcc.op.VectorToTensor;
import sibarum.mcc.op.VectorToTernion;
import sibarum.mcc.embedding.Embed;
import sibarum.mcc.embedding.Lookup;
import sibarum.mcc.embedding.SymbolEmbeddingTable;
import sibarum.mcc.op.advanced.SmoothedBasisElement;
import sibarum.mcc.op.block.MlpBlock;
import sibarum.mcc.op.block.TransformerBlock;

import sibarum.mcc.value.ValueType;

import java.util.List;
import java.util.Map;

/**
 * Static registry of mcc-core's built-in primitives. The importer
 * uses this to reconstruct primitives from their serialized names +
 * configs.
 *
 * <p>Stateless primitives ignore their config map. Parameterized
 * primitives read structural parameters (sizes, seed, etc.) from it.
 * Note: the {@code seed} a Trainable was constructed with affects
 * <em>initialization only</em>; after parameters are loaded from a
 * params blob, the seed is irrelevant.
 */
public final class BuiltinPrimitives {

    private BuiltinPrimitives() {}

    public static PrimitiveRegistry defaults() {
        PrimitiveRegistry r = new PrimitiveRegistry();

        // Stateless ops: ignore config.
        r.register("add", cfg -> new Add());
        r.register("sub", cfg -> new Sub());
        r.register("mul", cfg -> new Mul());
        r.register("dot-product", cfg -> new DotProduct());
        r.register("cosine-similarity", cfg -> new CosineSimilarity());
        r.register("similarity-gate", cfg -> new SimilarityGate());
        r.register("softmax", cfg -> new Softmax());
        r.register("magnitude", cfg -> new Magnitude());
        r.register("normalize", cfg -> new Normalize());
        r.register("cross-product-3", cfg -> new CrossProduct3());
        r.register("relu", cfg -> new Relu());
        r.register("sigmoid", cfg -> new Sigmoid());
        r.register("tanh", cfg -> new Tanh());
        r.register("concat", cfg -> new Concat());
        r.register("vector-to-int", cfg -> new VectorToInt());

        // Reshape primitives — pure shape conversion, no parameters.
        r.register("vector-to-ternion", cfg -> new VectorToTernion());
        r.register("ternion-to-vector", cfg -> new TernionToVector());
        r.register("vector-to-quaternion", cfg -> new VectorToQuaternion());
        r.register("quaternion-to-vector", cfg -> new QuaternionToVector());
        r.register("vector-to-tensor", cfg -> {
            @SuppressWarnings("unchecked")
            List<Number> rawShape = (List<Number>) requireField(cfg, "shape");
            int[] shape = rawShape.stream().mapToInt(Number::intValue).toArray();
            return new VectorToTensor(shape);
        });
        r.register("tensor-to-vector", cfg -> new TensorToVector());

        // Matrix ops on rank-2 TensorValues.
        r.register("matmul", cfg -> new MatMul());
        r.register("matvecmul", cfg -> new MatVecMul());

        // Slice has a structural config.
        r.register("slice", cfg -> {
            int from = ((Number) requireField(cfg, "from")).intValue();
            int to = ((Number) requireField(cfg, "to")).intValue();
            return new Slice(from, to);
        });

        // HarmonicLift: parameterless basis lift. Config carries K, inputDim,
        // kernel choice, and width fraction. No learned parameters to restore.
        r.register("harmonic-lift", cfg -> {
            int K = ((Number) requireField(cfg, "K")).intValue();
            int inputDim = ((Number) requireField(cfg, "inputDim")).intValue();
            SmoothedBasisElement.Kernel kernel = SmoothedBasisElement.Kernel.valueOf(
                    (String) requireField(cfg, "kernel"));
            double widthFrac = ((Number) cfg.getOrDefault("widthFrac", 0.0)).doubleValue();
            return new HarmonicLift(K, inputDim, kernel, widthFrac);
        });

        // Trainables: structural config selects shape; parameters are restored
        // from the params blob after construction.
        r.register("linear", cfg -> {
            int outDim = ((Number) requireField(cfg, "outDim")).intValue();
            int inDim = ((Number) requireField(cfg, "inDim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Linear(outDim, inDim, true, seed);
        });
        r.register("linear-no-bias", cfg -> {
            int outDim = ((Number) requireField(cfg, "outDim")).intValue();
            int inDim = ((Number) requireField(cfg, "inDim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Linear(outDim, inDim, false, seed);
        });
        r.register("int-to-vector", cfg -> {
            int vocabSize = ((Number) requireField(cfg, "vocabSize")).intValue();
            int dim = ((Number) requireField(cfg, "dim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new IntToVector(vocabSize, dim, seed);
        });

        // Parameter: no inputs, output-only learnable value. Config carries
        // the output type + shape so the registry rebuilds the right wrapper.
        r.register("parameter", cfg -> {
            ValueType type = ValueType.valueOf((String) requireField(cfg, "outputType"));
            @SuppressWarnings("unchecked")
            List<Number> rawShape = (List<Number>) cfg.getOrDefault("shape", List.of());
            int[] shape = rawShape.stream().mapToInt(Number::intValue).toArray();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Parameter(type, shape, seed);
        });

        r.register("mlp-block", cfg -> {
            @SuppressWarnings("unchecked")
            List<Number> rawSizes = (List<Number>) requireField(cfg, "sizes");
            int[] sizes = rawSizes.stream().mapToInt(Number::intValue).toArray();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new MlpBlock(sizes, seed);
        });
        r.register("transformer-block", cfg -> {
            int seqLen = ((Number) requireField(cfg, "seqLen")).intValue();
            int dIn = ((Number) requireField(cfg, "dIn")).intValue();
            int dModel = ((Number) requireField(cfg, "dModel")).intValue();
            int dFf = ((Number) requireField(cfg, "dFf")).intValue();
            int dOut = ((Number) requireField(cfg, "dOut")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new TransformerBlock(seqLen, dIn, dModel, dFf, dOut, seed);
        });

        // Embed: the config carries (dim, symbols-in-insertion-order); the
        // params blob carries the [N, dim] vector matrix. The factory must
        // pre-populate the table with the recorded symbols in the same order
        // so loadParameters' row-by-row overwrite lands on the right symbol.
        r.register("embed", cfg -> {
            int dim = ((Number) requireField(cfg, "dim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            @SuppressWarnings("unchecked")
            List<String> symbols = (List<String>) cfg.getOrDefault("symbols", List.of());
            SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, seed);
            for (String sym : symbols) table.embed(sym);
            return new Embed(table);
        });
        // Lookup: read-only over a pre-populated table. The runtime path
        // typically uses Embed's table; reconstructing Lookup directly is
        // not part of the MVP round-trip but the entry is included so
        // graphs that contain Lookup nodes can deserialize.
        r.register("lookup", cfg -> {
            int dim = ((Number) requireField(cfg, "dim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Lookup(new SymbolEmbeddingTable(dim, seed));
        });

        return r;
    }

    private static Object requireField(Map<String, Object> cfg, String key) {
        Object v = cfg.get(key);
        if (v == null) throw new IllegalArgumentException("primitive config missing field: " + key);
        return v;
    }
}
