package sibarum.mcc.graph;

import org.junit.jupiter.api.Test;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.graph.substrate.TransformationNode;
import sibarum.mcc.op.Add;
import sibarum.mcc.op.Mul;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GraphExecutionTest {

    private static final double TOL = 1e-12;

    @Test
    void executesThreeNodeChain() {
        // Substrate
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add1", new Add())
                .addNode("mul1", new Mul())
                .addNode("add2", new Add())
                .build();

        // Computation: result = (a + b) * c + d
        CompGraphNode n1 = new CompGraphNode("c-add1", tg.node("add1"));
        CompGraphNode n2 = new CompGraphNode("c-mul1", tg.node("mul1"));
        CompGraphNode n3 = new CompGraphNode("c-add2", tg.node("add2"));

        // Wire n1's output → n2 slot 0; n2's output → n3 slot 0.
        n2.wire(0, new SlotSource(n1, tg.edge(tg.node("add1"), tg.node("mul1"))));
        n3.wire(0, new SlotSource(n2, tg.edge(tg.node("mul1"), tg.node("add2"))));

        List<CompGraphNode> nodes = new ArrayList<>(List.of(n1, n2, n3));
        ComputationGraph cg = new ComputationGraph(nodes, n3);

        Value a = new MatrixValue(new double[] { 1, 2 });
        Value b = new MatrixValue(new double[] { 3, 4 });
        Value c = new MatrixValue(new double[] { 2, 2 });
        Value d = new MatrixValue(new double[] { 1, 1 });
        cg.bindRoot(n1, 0, a);
        cg.bindRoot(n1, 1, b);
        cg.bindRoot(n2, 1, c);
        cg.bindRoot(n3, 1, d);

        Value result = cg.execute();
        // (1+3)*2+1, (2+4)*2+1 = (9, 13)
        assertArrayEquals(new double[] { 9, 13 }, ((MatrixValue) result).data(), TOL);
    }

    @Test
    void detectsMissingRootBinding() {
        TransformationGraph tg = new TransformationGraphBuilder().addNode("add", new Add()).build();
        CompGraphNode n = new CompGraphNode("c-add", tg.node("add"));
        ComputationGraph cg = new ComputationGraph(List.of(n), n);
        cg.bindRoot(n, 0, new MatrixValue(new double[] { 1, 2 }));
        // slot 1 missing
        assertThrows(IllegalStateException.class, cg::execute);
    }

    @Test
    void reachesIsTransitive() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add1", new Add())
                .addNode("add2", new Add())
                .addNode("add3", new Add())
                .build();
        CompGraphNode n1 = new CompGraphNode("c-1", tg.node("add1"));
        CompGraphNode n2 = new CompGraphNode("c-2", tg.node("add2"));
        CompGraphNode n3 = new CompGraphNode("c-3", tg.node("add3"));
        n2.wire(0, new SlotSource(n1, tg.edge(tg.node("add1"), tg.node("add2"))));
        n3.wire(0, new SlotSource(n2, tg.edge(tg.node("add2"), tg.node("add3"))));
        ComputationGraph cg = new ComputationGraph(List.of(n1, n2, n3), n3);

        assertTrue(cg.reaches(n1, n3));
        assertTrue(cg.reaches(n2, n3));
        assertFalse(cg.reaches(n3, n1));
    }

    @Test
    void substrateEdgesAreTypeConsistent() {
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add", new Add())  // (MATRIX, MATRIX) -> MATRIX
                .addNode("mul", new Mul())  // (MATRIX, MATRIX) -> MATRIX
                .build();
        assertNotNull(tg.edge(tg.node("add"), tg.node("mul")));
        assertNotNull(tg.edge(tg.node("mul"), tg.node("add")));
    }
}
