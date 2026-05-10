package sibarum.strnn.demo;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.primitive.TreeOutputPrimitive;
import sibarum.strnn.rewrite.IdentityZero;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;

import java.util.ArrayList;
import java.util.List;

/**
 * v2 Phase 2: first symbolic rule running inside a ComputationGraph.
 *
 * Pipeline (manually carved):
 *   StringValue("5 + 0")  ->  ParseExpression  ->  IdentityZero  ->  TreeOutput
 * Mandates:
 *   intermediate: ((5 + 0)) tree must appear (the parsed unsimplified form)
 *   result:        Literal(5) (the simplified form)
 *
 * Confirms a rewrite primitive runs end-to-end through the v0 ComputationGraph
 * machinery and that mandate verification works on tree-shaped values.
 */
public final class IdentityZeroDemo {

    public static void main(String[] args) {
        TransformationGraph tg = buildTg();

        List<CompGraphNode> nodes = new ArrayList<>();
        int[] uid = {0};
        java.util.function.Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode parse = mk.apply(tg.node("parse_expr"));
        CompGraphNode rewrite = mk.apply(tg.node("identity_zero"));
        CompGraphNode out = mk.apply(tg.node("tree_output"));

        wire(tg, rewrite, 0, parse);
        wire(tg, out, 0, rewrite);

        ComputationGraph cg = new ComputationGraph(nodes, out);
        cg.bindRoot(parse, 0, new StringValue("5 + 0"));
        ParseTreeValue executed = (ParseTreeValue) cg.execute();
        System.out.println("input:  \"5 + 0\"");
        System.out.println("parsed: " + parse.producedValue());
        System.out.println("after rewrite: " + rewrite.producedValue());
        System.out.println("output: " + executed);

        ParseTreeValue parsedTree = ParseTreeValue.add(ParseTreeValue.lit(5), ParseTreeValue.lit(0));
        ParseTreeValue simplified = ParseTreeValue.lit(5);

        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("parsed_form", parsedTree, 0.0, 0),
                Mandate.result(simplified, 0.0, 1)));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        for (var e : report.outcomes().entrySet()) {
            String tag = e.getValue().satisfied() ? "OK   " : "FAIL ";
            String where = e.getValue().satisfied()
                    ? "@" + e.getValue().producedBy().id()
                    : "(" + e.getValue().reason() + ")";
            System.out.printf("    %s %-20s %s%n", tag, e.getKey().name(), where);
        }
        if (!report.allSatisfied()) {
            throw new AssertionError("Phase 2 verification failed");
        }
        System.out.println("\nv2 Phase 2 OK.");
    }

    private static TransformationGraph buildTg() {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("parse_expr", new ParseExpression());
        b.addNode("identity_zero", new IdentityZero());
        b.addNode("tree_output", new TreeOutputPrimitive());
        return b.build();
    }

    private static void wire(TransformationGraph tg, CompGraphNode target, int slot, CompGraphNode source) {
        TransformationEdge edge = tg.edge(source.tNode(), target.tNode());
        if (edge == null) {
            throw new IllegalStateException("no edge: " + source.tNode() + " -> " + target.tNode());
        }
        target.wire(slot, new SlotSource(source, edge));
    }
}
