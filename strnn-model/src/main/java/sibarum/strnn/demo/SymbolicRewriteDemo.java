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
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.TreeOutputPrimitive;
import sibarum.strnn.rewrite.Distribute;
import sibarum.strnn.rewrite.EvaluateBinaryOp;
import sibarum.strnn.rewrite.FactorCommon;
import sibarum.strnn.rewrite.IdentityOne;
import sibarum.strnn.rewrite.IdentityZero;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * v2 Phase 5: full primitive library + two-variant diagnostic demo.
 *
 * The library mixes pure-symbolic rewrite primitives (IdentityZero,
 * IdentityOne, Distribute, FactorCommon) with a learned-leaf primitive
 * (EvaluateBinaryOp), and a parser + tree output. Distribute and
 * FactorCommon are included for completeness but not exercised by these
 * variants — they would need sub-tree rule application (v3).
 *
 * Variant A — symbolic forced. Input "x + 0" (variable + literal zero)
 * with result mandate Variable("x"). EvaluateBinaryOp(ADD) cannot apply
 * because its pattern requires both children to be Literals (and one of
 * them is a Variable here); only IdentityZero applies (its pattern
 * {@code ?a + 0} accepts any subtree as ?a, including a Variable).
 *
 * Variant B — learned forced. Input "3 + 4" with result mandate Lit(7) at
 * tolerance 0.5. IdentityZero cannot apply because its pattern requires
 * a literal-zero operand; only EvaluateBinaryOp(ADD) produces an
 * approximate Lit(7) from two literal children.
 *
 * Pass criterion: in A the carved chain must contain identity-zero (and
 * not evaluate-add); in B it must contain evaluate-add (and not
 * identity-zero). If the carver picks correctly under both mandate sets,
 * the dual claim is empirically supported on the simplest possible test.
 */
public final class SymbolicRewriteDemo {

    public static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        TrainingDemoBootstrap.pretrainSilent(addMlp, MlpRole.ADD, 4000, 32, 0.01);
        TrainingDemoBootstrap.pretrainSilent(mulMlp, MlpRole.MUL, 12000, 32, 0.004);

        TransformationGraph tg = buildTg(addMlp, mulMlp);
        BackwardChainingCarver carver = new BackwardChainingCarver(/*seed=*/2024L, /*budget=*/300, /*epsilon=*/0.0);

        System.out.println("=== Variant A: symbolic forced (\"x + 0\", non-literal operand) ===");
        runVariant(carver, tg,
                "x + 0",
                ParseTreeValue.add(ParseTreeValue.var("x"), ParseTreeValue.lit(0)),
                ParseTreeValue.var("x"),
                /*resultTolerance=*/0.0,
                /*expectPrim=*/"identity-zero",
                /*forbidPrim=*/"evaluate-add");

        System.out.println("\n=== Variant B: learned forced (\"3 + 4\", no zero operand) ===");
        runVariant(carver, tg,
                "3 + 4",
                ParseTreeValue.add(ParseTreeValue.lit(3), ParseTreeValue.lit(4)),
                ParseTreeValue.lit(7),
                /*resultTolerance=*/0.5,
                /*expectPrim=*/"evaluate-add",
                /*forbidPrim=*/"identity-zero");

        System.out.println("\nv2 Phase 5: dual claim demonstrated on the simplest two-variant test.");
    }

    private static void runVariant(
            BackwardChainingCarver carver,
            TransformationGraph tg,
            String input,
            ParseTreeValue parsedForm,
            ParseTreeValue resultTree,
            double resultTolerance,
            String expectPrim,
            String forbidPrim) {

        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("parsed_form", parsedForm, 0.0, 0),
                Mandate.result(resultTree, resultTolerance, 1)));

        CarvingResult result = carver.carve(tg, mandates, new StringValue(input));
        if (result == null) {
            throw new AssertionError("carver returned null for input '" + input + "'");
        }
        result.graph().execute();

        Set<String> primNames = new HashSet<>();
        System.out.printf("  carved %d nodes, terminal = %s%n",
                result.graph().nodes().size(), result.graph().terminal().producedValue());
        for (CompGraphNode n : result.graph().topoOrder()) {
            Primitive p = n.tNode().primitive();
            primNames.add(p.name());
            System.out.printf("    %-30s = %s%n", n.id() + " [" + p.name() + "]",
                    n.producedValue());
        }

        VerificationReport report = new MandateVerifier().verify(result.graph(), mandates);
        for (var e : report.outcomes().entrySet()) {
            String tag = e.getValue().satisfied() ? "OK   " : "FAIL ";
            System.out.printf("    %s %-15s%n", tag, e.getKey().name());
        }
        if (!report.allSatisfied()) {
            throw new AssertionError("variant verification failed for '" + input + "'");
        }

        if (!primNames.contains(expectPrim)) {
            throw new AssertionError("expected primitive '" + expectPrim + "' in carving but got " + primNames);
        }
        if (primNames.contains(forbidPrim)) {
            throw new AssertionError("forbidden primitive '" + forbidPrim + "' appeared in carving");
        }
        System.out.printf("  PASS: '%s' present, '%s' absent%n", expectPrim, forbidPrim);
    }

    private static TransformationGraph buildTg(Mlp addMlp, Mlp mulMlp) {
        TransformationGraphBuilder b = new TransformationGraphBuilder();
        b.addNode("parse_expr", new ParseExpression());
        b.addNode("identity_zero", new IdentityZero());
        b.addNode("identity_one", new IdentityOne());
        b.addNode("distribute", new Distribute());
        b.addNode("factor_common", new FactorCommon());
        b.addNode("eval_add", new EvaluateBinaryOp(MlpRole.ADD, addMlp));
        b.addNode("eval_mul", new EvaluateBinaryOp(MlpRole.MUL, mulMlp));
        b.addNode("tree_output", new TreeOutputPrimitive());
        return b.build();
    }
}
