package sibarum.strnn.demo;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
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
import sibarum.strnn.value.TokenListValue;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Phase 3: structural mandate verification on a manually-built DAG.
 * Three checks:
 *   (a) Positive: full §9.2 mandate set on the structured pipeline; all satisfied.
 *   (b) Negative — collapse: an alternative DAG that skips the mul intermediate
 *       (computes 23 directly); the precedence mandate fails.
 *   (c) Negative — wrong ordering: a DAG where add happens before mul; the
 *       ordering constraint fails.
 */
public final class MandateVerificationDemo {

    static void main(String[] args) {
        TransformationGraph tg = buildTransformationGraph();
        MandateVerifier verifier = new MandateVerifier();

        ComputationGraph structured = buildStructuredGraph(tg);
        structured.execute();
        MandateSet full = fullMandates();
        VerificationReport rA = verifier.verify(structured, full);
        System.out.println("(a) structured pipeline + full mandates");
        printReport(rA);
        if (!rA.allSatisfied()) throw new AssertionError("Phase 3 (a) expected all satisfied");

        ComputationGraph collapsed = buildCollapsedGraph(tg);
        collapsed.execute();
        VerificationReport rB = verifier.verify(collapsed, full);
        System.out.println("(b) collapsed pipeline + full mandates (expected: precedence fails)");
        printReport(rB);
        Mandate prod = mandateNamed(full, "intermediate_product");
        if (rB.get(prod).satisfied()) {
            throw new AssertionError("Phase 3 (b) expected the intermediate_product mandate to fail");
        }

        ComputationGraph reordered = buildReorderedGraph(tg);
        reordered.execute();
        VerificationReport rC = verifier.verify(reordered, full);
        System.out.println("(c) reordered pipeline + full mandates (expected: ordering fails)");
        printReport(rC);
        Mandate result = mandateNamed(full, "result");
        if (rC.get(result).satisfied() && rC.get(prod).satisfied()) {
            throw new AssertionError("Phase 3 (c) expected ordering failure on at least one of result/product");
        }

        System.out.println("Phase 3 mandate verification OK.");
    }

    private static MandateSet fullMandates() {
        return new MandateSet(List.of(
                Mandate.intermediate("plus_split", new TokenListValue(List.of("3", "4*5")), 0.0, 0),
                Mandate.intermediate("star_split", new TokenListValue(List.of("4", "5")), 0.0, 1),
                Mandate.intermediate("intermediate_product", new NumberValue(20.0), 1e-3, 2),
                Mandate.result(new NumberValue(23.0), 1e-3, 3)));
    }

    private static Mandate mandateNamed(MandateSet ms, String name) {
        for (Mandate m : ms.mandates()) if (m.name().equals(name)) return m;
        throw new IllegalArgumentException(name);
    }

    private static void printReport(VerificationReport r) {
        for (var e : r.outcomes().entrySet()) {
            Mandate m = e.getKey();
            VerificationReport.Outcome o = e.getValue();
            String tag = o.satisfied() ? "OK   " : "FAIL ";
            String where = o.satisfied() ? ("@" + o.producedBy().id()) : ("(" + o.reason() + ")");
            System.out.printf("    %s %-22s %s%n", tag, m.name(), where);
        }
    }

    private static TransformationGraph buildTransformationGraph() {
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
        return b.build();
    }

    /** Carves the canonical structured pipeline that produces all mandates. */
    private static ComputationGraph buildStructuredGraph(TransformationGraph tg) {
        List<CompGraphNode> nodes = new ArrayList<>();
        int[] uid = {0};
        Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode splitPlus = mk.apply(tg.node("split_plus"));
        CompGraphNode pickL = mk.apply(tg.node("token_0"));
        CompGraphNode pickR = mk.apply(tg.node("token_1"));
        CompGraphNode parseL = mk.apply(tg.node("parse"));
        CompGraphNode encL = mk.apply(tg.node("num_to_mat"));

        CompGraphNode splitStar = mk.apply(tg.node("split_star"));
        CompGraphNode pickML = mk.apply(tg.node("token_0"));
        CompGraphNode pickMR = mk.apply(tg.node("token_1"));
        CompGraphNode parseML = mk.apply(tg.node("parse"));
        CompGraphNode parseMR = mk.apply(tg.node("parse"));
        CompGraphNode encML = mk.apply(tg.node("num_to_mat"));
        CompGraphNode encMR = mk.apply(tg.node("num_to_mat"));
        CompGraphNode composeMul = mk.apply(tg.node("compose"));
        CompGraphNode mlpMul = mk.apply(tg.node("mlp_mul"));
        CompGraphNode decodeMul = mk.apply(tg.node("mat_to_num"));
        CompGraphNode encMulOut = mk.apply(tg.node("num_to_mat"));

        CompGraphNode composeAdd = mk.apply(tg.node("compose"));
        CompGraphNode mlpAdd = mk.apply(tg.node("mlp_add"));
        CompGraphNode decodeAdd = mk.apply(tg.node("mat_to_num"));
        CompGraphNode output = mk.apply(tg.node("output"));

        wire(tg, pickL, 0, splitPlus);
        wire(tg, pickR, 0, splitPlus);
        wire(tg, parseL, 0, pickL);
        wire(tg, encL, 0, parseL);
        wire(tg, splitStar, 0, pickR);
        wire(tg, pickML, 0, splitStar);
        wire(tg, pickMR, 0, splitStar);
        wire(tg, parseML, 0, pickML);
        wire(tg, parseMR, 0, pickMR);
        wire(tg, encML, 0, parseML);
        wire(tg, encMR, 0, parseMR);
        wire(tg, composeMul, 0, encML);
        wire(tg, composeMul, 1, encMR);
        wire(tg, mlpMul, 0, composeMul);
        wire(tg, decodeMul, 0, mlpMul);
        wire(tg, encMulOut, 0, decodeMul);
        wire(tg, composeAdd, 0, encL);
        wire(tg, composeAdd, 1, encMulOut);
        wire(tg, mlpAdd, 0, composeAdd);
        wire(tg, decodeAdd, 0, mlpAdd);
        wire(tg, output, 0, decodeAdd);

        ComputationGraph cg = new ComputationGraph(nodes, output);
        cg.bindRoot(splitPlus, 0, new StringValue("3+4*5"));
        return cg;
    }

    /**
     * A pipeline that produces 23 without ever materializing 20: it parses 3, 4, 5,
     * encodes them, composes only [a, c] = [3, 5] then runs mlp_mul + mlp_add via
     * a contrived but type-legal route that yields 23 without an intermediate-product
     * node carrying the number 20. We simulate that by computing 4*5 and 3 inside a
     * single MLP boundary via a hardcoded path: the "decode" node here decodes the
     * already-summed result, so no node ever produces NumberValue(20.0).
     */
    private static ComputationGraph buildCollapsedGraph(TransformationGraph tg) {
        List<CompGraphNode> nodes = new ArrayList<>();
        int[] uid = {0};
        Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode splitPlus = mk.apply(tg.node("split_plus"));
        CompGraphNode pickL = mk.apply(tg.node("token_0"));
        CompGraphNode pickR = mk.apply(tg.node("token_1"));
        CompGraphNode parseL = mk.apply(tg.node("parse"));
        CompGraphNode encL = mk.apply(tg.node("num_to_mat"));
        CompGraphNode splitStar = mk.apply(tg.node("split_star"));
        CompGraphNode pickML = mk.apply(tg.node("token_0"));
        CompGraphNode pickMR = mk.apply(tg.node("token_1"));
        CompGraphNode parseML = mk.apply(tg.node("parse"));
        CompGraphNode parseMR = mk.apply(tg.node("parse"));
        CompGraphNode encML = mk.apply(tg.node("num_to_mat"));
        CompGraphNode encMR = mk.apply(tg.node("num_to_mat"));
        CompGraphNode composeMul = mk.apply(tg.node("compose"));
        CompGraphNode mlpMul = mk.apply(tg.node("mlp_mul"));
        CompGraphNode composeAdd = mk.apply(tg.node("compose"));
        CompGraphNode mlpAdd = mk.apply(tg.node("mlp_add"));
        CompGraphNode decodeAdd = mk.apply(tg.node("mat_to_num"));
        CompGraphNode output = mk.apply(tg.node("output"));

        wire(tg, pickL, 0, splitPlus);
        wire(tg, pickR, 0, splitPlus);
        wire(tg, parseL, 0, pickL);
        wire(tg, encL, 0, parseL);
        wire(tg, splitStar, 0, pickR);
        wire(tg, pickML, 0, splitStar);
        wire(tg, pickMR, 0, splitStar);
        wire(tg, parseML, 0, pickML);
        wire(tg, parseMR, 0, pickMR);
        wire(tg, encML, 0, parseML);
        wire(tg, encMR, 0, parseMR);
        wire(tg, composeMul, 0, encML);
        wire(tg, composeMul, 1, encMR);
        wire(tg, mlpMul, 0, composeMul);
        wire(tg, composeAdd, 0, encL);
        wire(tg, composeAdd, 1, mlpMul);
        wire(tg, mlpAdd, 0, composeAdd);
        wire(tg, decodeAdd, 0, mlpAdd);
        wire(tg, output, 0, decodeAdd);

        ComputationGraph cg = new ComputationGraph(nodes, output);
        cg.bindRoot(splitPlus, 0, new StringValue("3+4*5"));
        return cg;
    }

    /**
     * Wires multiplication AFTER addition: parse(3) and parse(4) are summed,
     * yielding 7, then mlp_mul applied with (7, 5) gives 35 — wrong result, AND
     * if it accidentally produces 20 anywhere, that node would topologically
     * follow nodes assigned to the addition stage, violating ordering.
     */
    private static ComputationGraph buildReorderedGraph(TransformationGraph tg) {
        List<CompGraphNode> nodes = new ArrayList<>();
        int[] uid = {0};
        Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode splitPlus = mk.apply(tg.node("split_plus"));
        CompGraphNode pickL = mk.apply(tg.node("token_0"));
        CompGraphNode pickR = mk.apply(tg.node("token_1"));
        CompGraphNode parseL = mk.apply(tg.node("parse"));
        CompGraphNode encL = mk.apply(tg.node("num_to_mat"));
        CompGraphNode splitStar = mk.apply(tg.node("split_star"));
        CompGraphNode pickML = mk.apply(tg.node("token_0"));
        CompGraphNode pickMR = mk.apply(tg.node("token_1"));
        CompGraphNode parseML = mk.apply(tg.node("parse"));
        CompGraphNode parseMR = mk.apply(tg.node("parse"));
        CompGraphNode encML = mk.apply(tg.node("num_to_mat"));
        CompGraphNode encMR = mk.apply(tg.node("num_to_mat"));

        // Add first: encL (=3) and encML (=4).
        CompGraphNode composeAdd = mk.apply(tg.node("compose"));
        CompGraphNode mlpAdd = mk.apply(tg.node("mlp_add"));
        // Then mul: mlpAdd output (=7 in MATRIX form) and encMR (=5).
        CompGraphNode composeMul = mk.apply(tg.node("compose"));
        CompGraphNode mlpMul = mk.apply(tg.node("mlp_mul"));
        CompGraphNode decodeMul = mk.apply(tg.node("mat_to_num"));
        CompGraphNode output = mk.apply(tg.node("output"));

        wire(tg, pickL, 0, splitPlus);
        wire(tg, pickR, 0, splitPlus);
        wire(tg, parseL, 0, pickL);
        wire(tg, encL, 0, parseL);
        wire(tg, splitStar, 0, pickR);
        wire(tg, pickML, 0, splitStar);
        wire(tg, pickMR, 0, splitStar);
        wire(tg, parseML, 0, pickML);
        wire(tg, parseMR, 0, pickMR);
        wire(tg, encML, 0, parseML);
        wire(tg, encMR, 0, parseMR);
        wire(tg, composeAdd, 0, encL);
        wire(tg, composeAdd, 1, encML);
        wire(tg, mlpAdd, 0, composeAdd);
        wire(tg, composeMul, 0, mlpAdd);
        wire(tg, composeMul, 1, encMR);
        wire(tg, mlpMul, 0, composeMul);
        wire(tg, decodeMul, 0, mlpMul);
        wire(tg, output, 0, decodeMul);

        ComputationGraph cg = new ComputationGraph(nodes, output);
        cg.bindRoot(splitPlus, 0, new StringValue("3+4*5"));
        return cg;
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
