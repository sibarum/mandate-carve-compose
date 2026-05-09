package sibarum.strnn.demo;

import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
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
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;
import sibarum.strnn.value.Value;

import java.util.List;

/**
 * Phase 4: BackwardChainingCarver. Construct the computation graph
 * automatically from mandates, execute, verify.
 */
public final class CarvingDemo {
    static void main(String[] args) {
        TransformationGraph tg = buildTransformationGraph();
        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("plus_split",
                        new TokenListValue(List.of("3", "4*5")), 0.0, 0),
                Mandate.intermediate("star_split",
                        new TokenListValue(List.of("4", "5")), 0.0, 1),
                Mandate.intermediate("intermediate_product",
                        new NumberValue(20.0), 1e-3, 2),
                Mandate.result(new NumberValue(23.0), 1e-3, 3)));

        BackwardChainingCarver carver = new BackwardChainingCarver(123L);
        CarvingResult result = carver.carve(tg, mandates, new StringValue("3+4*5"));
        if (result == null) {
            throw new AssertionError("carver returned null");
        }

        System.out.printf("carved graph: %d nodes%n", result.graph().nodes().size());
        Value out = result.graph().execute();
        System.out.println("executed result: " + out);

        VerificationReport report = new MandateVerifier().verify(result.graph(), mandates);
        System.out.printf("simulated values tracked: %d%n", result.simulatedValues().size());
        for (var e : report.outcomes().entrySet()) {
            Mandate m = e.getKey();
            VerificationReport.Outcome o = e.getValue();
            String tag = o.satisfied() ? "OK   " : "FAIL ";
            String where = o.satisfied() ? ("@" + o.producedBy().id()) : ("(" + o.reason() + ")");
            System.out.printf("    %s %-22s %s%n", tag, m.name(), where);
        }

        if (!report.allSatisfied()) {
            throw new AssertionError("Phase 4: carver did not satisfy all mandates");
        }
        System.out.println("\nTopo order:");
        for (CompGraphNode n : result.graph().topoOrder()) {
            System.out.printf("  %-30s = %s%n", n.id(), n.producedValue());
        }
        System.out.println("Phase 4 carving OK.");
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
}
