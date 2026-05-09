package sibarum.strnn.demo;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.MlpPrimitive;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

/**
 * Phase 2: build the any-to-any-modulo-types TransformationGraph from the
 * §9.2 primitive library, then manually stitch a ComputationGraph that
 * evaluates &quot;3+4*5&quot;. Confirms the graph types and the executor work
 * end-to-end before any carving exists.
 */
public final class ManualGraphDemo {
    static void main(String[] args) {
        // Build the transformation graph (any-to-any modulo type compatibility).
        // MLPs are stub mode here so this demo doesn't need pre-trained weights.
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("split_plus", new SplitStringAt('+'));
        b.addNode("split_star", new SplitStringAt('*'));
        b.addNode("token_0", new TokenAt(0));
        b.addNode("token_1", new TokenAt(1));
        b.addNode("parse", new ParseNumber());
        b.addNode("num_to_mat", new NumberToMatrix());
        b.addNode("compose", new ComposeMatrices());
        b.addNode("mlp_add", new MlpPrimitive(MlpRole.ADD));
        b.addNode("mlp_mul", new MlpPrimitive(MlpRole.MUL));
        b.addNode("mat_to_num", new MatrixToNumber());
        b.addNode("output", new OutputPrimitive());
        TransformationGraph tg = b.build();

        System.out.printf("transformation graph: %d nodes, %d edges%n",
                tg.nodes().size(), tg.edges().size());

        // Manually carve a computation graph for "3+4*5".
        java.util.List<CompGraphNode> nodes = new java.util.ArrayList<>();
        int[] uid = {0};
        java.util.function.Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode splitPlus = mk.apply(tg.node("split_plus"));
        CompGraphNode pickLeft = mk.apply(tg.node("token_0"));
        CompGraphNode pickRight = mk.apply(tg.node("token_1"));
        CompGraphNode parseLeft = mk.apply(tg.node("parse"));
        CompGraphNode encLeft = mk.apply(tg.node("num_to_mat"));

        CompGraphNode splitStar = mk.apply(tg.node("split_star"));
        CompGraphNode pickMulLeft = mk.apply(tg.node("token_0"));
        CompGraphNode pickMulRight = mk.apply(tg.node("token_1"));
        CompGraphNode parseMulLeft = mk.apply(tg.node("parse"));
        CompGraphNode parseMulRight = mk.apply(tg.node("parse"));
        CompGraphNode encMulLeft = mk.apply(tg.node("num_to_mat"));
        CompGraphNode encMulRight = mk.apply(tg.node("num_to_mat"));
        CompGraphNode composeMul = mk.apply(tg.node("compose"));
        CompGraphNode mlpMul = mk.apply(tg.node("mlp_mul"));
        CompGraphNode decodeMul = mk.apply(tg.node("mat_to_num"));
        CompGraphNode encMulOut = mk.apply(tg.node("num_to_mat"));

        CompGraphNode composeAdd = mk.apply(tg.node("compose"));
        CompGraphNode mlpAdd = mk.apply(tg.node("mlp_add"));
        CompGraphNode decodeAdd = mk.apply(tg.node("mat_to_num"));
        CompGraphNode output = mk.apply(tg.node("output"));

        wire(tg, pickLeft, 0, splitPlus);
        wire(tg, pickRight, 0, splitPlus);
        wire(tg, parseLeft, 0, pickLeft);
        wire(tg, encLeft, 0, parseLeft);

        wire(tg, splitStar, 0, pickRight);
        wire(tg, pickMulLeft, 0, splitStar);
        wire(tg, pickMulRight, 0, splitStar);
        wire(tg, parseMulLeft, 0, pickMulLeft);
        wire(tg, parseMulRight, 0, pickMulRight);
        wire(tg, encMulLeft, 0, parseMulLeft);
        wire(tg, encMulRight, 0, parseMulRight);
        wire(tg, composeMul, 0, encMulLeft);
        wire(tg, composeMul, 1, encMulRight);
        wire(tg, mlpMul, 0, composeMul);
        wire(tg, decodeMul, 0, mlpMul);
        wire(tg, encMulOut, 0, decodeMul);

        wire(tg, composeAdd, 0, encLeft);
        wire(tg, composeAdd, 1, encMulOut);
        wire(tg, mlpAdd, 0, composeAdd);
        wire(tg, decodeAdd, 0, mlpAdd);
        wire(tg, output, 0, decodeAdd);

        ComputationGraph cg = new ComputationGraph(nodes, output);
        cg.bindRoot(splitPlus, 0, new StringValue("3+4*5"));

        Value result = cg.execute();
        double answer = ((NumberValue) result).n();
        System.out.println("manual carving produced: " + answer);
        if (Math.abs(answer - 23.0) > 1e-6) {
            throw new AssertionError("expected 23.0, got " + answer);
        }

        System.out.println("intermediate at decodeMul: " + decodeMul.producedValue());
        System.out.println("Phase 2 manual graph OK.");
    }

    private static void wire(TransformationGraph tg, CompGraphNode target, int slot, CompGraphNode source) {
        TransformationEdge edge = tg.edge(source.tNode(), target.tNode());
        if (edge == null) {
            throw new IllegalStateException(
                    "no transformation edge: " + source.tNode() + " -> " + target.tNode());
        }
        target.wire(slot, new SlotSource(source, edge));
    }
}
