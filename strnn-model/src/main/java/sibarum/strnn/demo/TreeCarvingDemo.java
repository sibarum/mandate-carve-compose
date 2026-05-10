package sibarum.strnn.demo;

import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.primitive.TreeOutputPrimitive;
import sibarum.strnn.rewrite.EvaluateBinaryOp;
import sibarum.strnn.rewrite.IdentityZero;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;

import java.util.List;

/**
 * v2 Phase 4: carver-driven construction of a tree-shaped pipeline.
 *
 * Input: "5 + 0"
 * Mandates:
 *   parsed_form: BinaryOp(+, Lit(5), Lit(0)) must appear (forces parsing)
 *   result:      Lit(5) (forces simplification — IdentityZero wins over
 *                EvaluateBinaryOp here because IdentityZero requires a Lit(0)
 *                operand which only the parsed form supplies)
 *
 * Expected carved chain: parse_expr -&gt; identity_zero -&gt; tree_output.
 */
public final class TreeCarvingDemo {

    public static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);

        TransformationGraph tg = buildTg(addMlp, mulMlp);
        BackwardChainingCarver carver = new BackwardChainingCarver(123L);

        ParseTreeValue parsedTree = ParseTreeValue.add(ParseTreeValue.lit(5), ParseTreeValue.lit(0));
        ParseTreeValue simplified = ParseTreeValue.lit(5);
        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("parsed_form", parsedTree, 0.0, 0),
                Mandate.result(simplified, 0.5, 1)));  // tolerance for MLP-approximate literals

        CarvingResult result = carver.carve(tg, mandates, new StringValue("5 + 0"));
        if (result == null) {
            throw new AssertionError("carver returned null");
        }

        System.out.printf("carved %d nodes%n", result.graph().nodes().size());
        result.graph().execute();

        System.out.println("Topo order:");
        for (CompGraphNode n : result.graph().topoOrder()) {
            System.out.printf("  %-30s = %s%n", n.id(), n.producedValue());
        }

        VerificationReport report = new MandateVerifier().verify(result.graph(), mandates);
        for (var e : report.outcomes().entrySet()) {
            String tag = e.getValue().satisfied() ? "OK   " : "FAIL ";
            String where = e.getValue().satisfied()
                    ? "@" + e.getValue().producedBy().id()
                    : "(" + e.getValue().reason() + ")";
            System.out.printf("    %s %-20s %s%n", tag, e.getKey().name(), where);
        }
        if (!report.allSatisfied()) {
            throw new AssertionError("Phase 4 verification failed");
        }
        System.out.println("\nv2 Phase 4 OK.");
    }

    private static TransformationGraph buildTg(Mlp addMlp, Mlp mulMlp) {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("parse_expr", new ParseExpression());
        b.addNode("identity_zero", new IdentityZero());
        b.addNode("eval_add", new EvaluateBinaryOp(MlpRole.ADD, addMlp));
        b.addNode("eval_mul", new EvaluateBinaryOp(MlpRole.MUL, mulMlp));
        b.addNode("tree_output", new TreeOutputPrimitive());
        return b.build();
    }
}
