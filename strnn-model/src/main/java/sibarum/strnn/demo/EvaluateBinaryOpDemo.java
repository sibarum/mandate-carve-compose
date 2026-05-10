package sibarum.strnn.demo;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.primitive.TreeOutputPrimitive;
import sibarum.strnn.rewrite.EvaluateBinaryOp;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * v2 Phase 3: heterogeneity primitive end-to-end. EvaluateBinaryOp wraps a
 * pretrained MLP inside a symbolic rule (Lit op Lit -&gt; Lit) and lives in
 * the transformation graph alongside pure-symbolic rules.
 *
 * Demo: parse "3 + 4" -&gt; BinaryOp(+, Lit(3), Lit(4)) -&gt; evaluate-add
 * -&gt; Lit(7) (approximately, given MLP accuracy). Verifies the learned
 * leaf produces a reasonable result and the chain executes.
 */
public final class EvaluateBinaryOpDemo {
    public static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);

        TransformationGraph tg = buildTg(addMlp, mulMlp);

        // Run "3 + 4" through evaluate-add at root.
        List<CompGraphNode> nodes = new ArrayList<>();
        int[] uid = {0};
        Function<TransformationNode, CompGraphNode> mk = t -> {
            CompGraphNode n = new CompGraphNode("c" + (uid[0]++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        };

        CompGraphNode parse = mk.apply(tg.node("parse_expr"));
        CompGraphNode evalAdd = mk.apply(tg.node("eval_add"));
        CompGraphNode out = mk.apply(tg.node("tree_output"));
        wire(tg, evalAdd, 0, parse);
        wire(tg, out, 0, evalAdd);

        ComputationGraph cg = new ComputationGraph(nodes, out);
        cg.bindRoot(parse, 0, new StringValue("3 + 4"));
        ParseTreeValue executed = (ParseTreeValue) cg.execute();
        System.out.println("\"3 + 4\" -> parsed: " + parse.producedValue() + "  eval: " + executed);

        // Now "6 * 7"
        nodes.clear();
        uid[0] = 0;
        CompGraphNode parse2 = mk.apply(tg.node("parse_expr"));
        CompGraphNode evalMul = mk.apply(tg.node("eval_mul"));
        CompGraphNode out2 = mk.apply(tg.node("tree_output"));
        wire(tg, evalMul, 0, parse2);
        wire(tg, out2, 0, evalMul);
        ComputationGraph cg2 = new ComputationGraph(nodes, out2);
        cg2.bindRoot(parse2, 0, new StringValue("6 * 7"));
        ParseTreeValue executed2 = (ParseTreeValue) cg2.execute();
        System.out.println("\"6 * 7\" -> parsed: " + parse2.producedValue() + "  eval: " + executed2);

        // Mandate: result is Lit(7) within tolerance? The verifier on tree values uses exact match;
        // here we manually check tolerance because MLP output is approximate.
        double r1 = ((ParseTreeValue.Literal) executed).value();
        double r2 = ((ParseTreeValue.Literal) executed2).value();
        if (Math.abs(r1 - 7.0) > 0.5) throw new AssertionError("eval-add far from 7: " + r1);
        if (Math.abs(r2 - 42.0) > 1.0) throw new AssertionError("eval-mul far from 42: " + r2);

        // Try mandate verification with a parsed-form intermediate
        MandateSet ms = new MandateSet(List.of(
                Mandate.intermediate("parsed_form",
                        ParseTreeValue.add(ParseTreeValue.lit(3), ParseTreeValue.lit(4)), 0.0, 0)));
        VerificationReport report = new MandateVerifier().verify(cg, ms);
        if (!report.allSatisfied()) {
            throw new AssertionError("expected parsed_form mandate to be satisfied");
        }
        System.out.println("intermediate parsed_form mandate satisfied for first carving.");

        System.out.println("\nv2 Phase 3 OK.");
    }

    private static TransformationGraph buildTg(Mlp addMlp, Mlp mulMlp) {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("parse_expr", new ParseExpression());
        b.addNode("eval_add", new EvaluateBinaryOp(MlpRole.ADD, addMlp));
        b.addNode("eval_mul", new EvaluateBinaryOp(MlpRole.MUL, mulMlp));
        b.addNode("tree_output", new TreeOutputPrimitive());
        return b.build();
    }

    private static void wire(TransformationGraph tg, CompGraphNode target, int slot, CompGraphNode source) {
        TransformationEdge edge = tg.edge(source.tNode(), target.tNode());
        if (edge == null) throw new IllegalStateException("no edge: " + source.tNode() + " -> " + target.tNode());
        target.wire(slot, new SlotSource(source, edge));
    }
}
