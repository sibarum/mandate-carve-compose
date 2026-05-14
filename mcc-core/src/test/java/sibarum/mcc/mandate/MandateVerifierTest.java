package sibarum.mcc.mandate;

import org.junit.jupiter.api.Test;
import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.Add;
import sibarum.mcc.op.Mul;
import sibarum.mcc.value.MatrixValue;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MandateVerifierTest {

    @Test
    void satisfiedResultMandate() {
        ComputationGraph cg = buildAddMulChain();
        execute(cg);

        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new MatrixValue(new double[] { 9, 13 }), 1e-9, 10)
        ));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        assertTrue(report.allSatisfied());
    }

    @Test
    void failingResultMandate() {
        ComputationGraph cg = buildAddMulChain();
        execute(cg);

        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new MatrixValue(new double[] { 0, 0 }), 1e-9, 10)
        ));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        assertFalse(report.allSatisfied());
        assertEquals(1, report.total());
        assertEquals(0, report.satisfiedCount());
    }

    @Test
    void intermediateMandateMatchesAnyNode() {
        ComputationGraph cg = buildAddMulChain();
        execute(cg);

        // The add step produces (1+3, 2+4) = (4, 6)
        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("sum", new MatrixValue(new double[] { 4, 6 }), 1e-9, 1),
                Mandate.result(new MatrixValue(new double[] { 9, 13 }), 1e-9, 10)
        ));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        assertTrue(report.allSatisfied());
    }

    @Test
    void orderingViolationDetected() {
        // Two add nodes, NOT wired to each other. n1 and n2 both produce (4, 6)
        // (n1 by chance, n2 because its inputs were bound to a value matching
        // the intermediate mandate). With ordering, the verifier looks for the
        // intermediate mandate first — finds n1 (lower order in nodes()), then
        // checks whether n1 reaches the terminal n2. It does NOT (they're
        // disconnected), so the ordering check must fire.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add1", new Add())
                .addNode("add2", new Add())
                .build();
        CompGraphNode n1 = new CompGraphNode("c-add1", tg.node("add1"));
        CompGraphNode n2 = new CompGraphNode("c-add2", tg.node("add2"));
        ComputationGraph cg = new ComputationGraph(List.of(n1, n2), n2);

        cg.bindRoot(n1, 0, new MatrixValue(new double[] { 1, 2 }));
        cg.bindRoot(n1, 1, new MatrixValue(new double[] { 3, 4 }));   // n1 → (4, 6)
        cg.bindRoot(n2, 0, new MatrixValue(new double[] { 4, 6 }));
        cg.bindRoot(n2, 1, new MatrixValue(new double[] { 0, 0 }));   // n2 → (4, 6)
        cg.execute();

        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("sum", new MatrixValue(new double[] { 4, 6 }), 1e-9, 1),
                Mandate.result(new MatrixValue(new double[] { 4, 6 }), 1e-9, 10)
        ));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        assertFalse(report.allSatisfied(),
                "Ordering violation must fire when intermediate's producer doesn't reach terminal");

        // Specifically, the result mandate should be the one re-marked failed
        // by the ordering check.
        Mandate result = mandates.result();
        VerificationReport.Outcome resultOutcome = report.get(result);
        assertFalse(resultOutcome.satisfied());
        assertNotNull(resultOutcome.reason());
        assertTrue(resultOutcome.reason().contains("ordering violation"),
                "expected ordering violation message, got: " + resultOutcome.reason());
    }

    @Test
    void intermediateMandateMissingFails() {
        ComputationGraph cg = buildAddMulChain();
        execute(cg);

        MandateSet mandates = new MandateSet(List.of(
                Mandate.intermediate("does-not-exist",
                        new MatrixValue(new double[] { 999, 999 }), 1e-9, 1)
        ));
        VerificationReport report = new MandateVerifier().verify(cg, mandates);
        assertFalse(report.allSatisfied());
        VerificationReport.Outcome outcome = report.outcomes().values().iterator().next();
        assertNotNull(outcome.reason());
    }

    private static ComputationGraph buildAddMulChain() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add", new Add())
                .addNode("mul", new Mul())
                .addNode("add2", new Add())
                .build();
        CompGraphNode n1 = new CompGraphNode("c-add", tg.node("add"));
        CompGraphNode n2 = new CompGraphNode("c-mul", tg.node("mul"));
        CompGraphNode n3 = new CompGraphNode("c-add2", tg.node("add2"));
        n2.wire(0, new SlotSource(n1, tg.edge(tg.node("add"), tg.node("mul"))));
        n3.wire(0, new SlotSource(n2, tg.edge(tg.node("mul"), tg.node("add2"))));
        return new ComputationGraph(List.of(n1, n2, n3), n3);
    }

    private static void execute(ComputationGraph cg) {
        // Look up the nodes by id to bind roots
        for (CompGraphNode n : cg.nodes()) {
            if (n.id().equals("c-add")) {
                cg.bindRoot(n, 0, new MatrixValue(new double[] { 1, 2 }));
                cg.bindRoot(n, 1, new MatrixValue(new double[] { 3, 4 }));
            } else if (n.id().equals("c-mul")) {
                cg.bindRoot(n, 1, new MatrixValue(new double[] { 2, 2 }));
            } else if (n.id().equals("c-add2")) {
                cg.bindRoot(n, 1, new MatrixValue(new double[] { 1, 1 }));
            }
        }
        cg.execute();
    }
}
