package sibarum.strnn.carving;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.LearnedArithmetic;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.primitive.TreeOutputPrimitive;
import sibarum.strnn.rewrite.EvaluateBinaryOp;
import sibarum.strnn.rewrite.RewriteRulePrimitive;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueDistance;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * v0 carver per plan §3: backward chaining from the result mandate, weighted
 * by edge stats, with mandate values used as concrete waypoints.
 *
 * The algorithm:
 *   1. Create the terminal node from an OutputPrimitive transformation node.
 *   2. Solve(target = result mandate's expected value, attached to terminal slot 0).
 *   3. solve() considers options in this order: (a) wire to root; (b) wire to
 *      an existing node whose simulated value matches; (c) create a new node by
 *      picking a TransformationNode whose primitive can produce the target.
 *   4. For (c), the carver inverts the candidate primitive — given the desired
 *      output value, it computes the input value(s) that would produce it. For
 *      MlpPrimitive that inversion is non-unique; the carver enumerates pairs
 *      drawn from the mandate value pool to break the ambiguity.
 *   5. Candidates are sorted by edge-stats score (incoming edge from candidate
 *      to the parent), high score first; ties broken randomly.
 *   6. Budget caps overall recursion. On budget exhaustion the carving fails;
 *      the trainer records the partial trace and updates edge stats accordingly.
 *
 * §6.1 non-locality is preserved: solve() never pins an intermediate mandate to
 * a fixed location. It is the verifier (run by the trainer) that decides
 * whether a carving's executed values satisfy each mandate.
 */
public final class BackwardChainingCarver {
    private static final int DEFAULT_BUDGET = 200;
    private static final int MAX_DEPTH = 16;

    private final Random rng;
    private final int budget;
    private final double epsilon;

    public BackwardChainingCarver(long seed) {
        this(seed, DEFAULT_BUDGET, 0.0);
    }

    public BackwardChainingCarver(long seed, int budget) {
        this(seed, budget, 0.0);
    }

    /**
     * @param epsilon exploration rate. With probability epsilon a candidate
     *                ranking step skips the score-based sort and uses random
     *                order instead, letting alternative edges accumulate
     *                samples instead of locking in to whichever edge won the
     *                initial coin flip. Default 0.0 preserves v0 behavior.
     */
    public BackwardChainingCarver(long seed, int budget, double epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("epsilon must be in [0, 1]");
        }
        this.rng = new Random(seed);
        this.budget = budget;
        this.epsilon = epsilon;
    }

    public CarvingResult carve(TransformationGraph tg, MandateSet mandates, Value rootInput) {
        Mandate result = mandates.result();
        if (result == null) {
            throw new IllegalArgumentException("MandateSet has no result mandate");
        }

        TransformationNode outputTNode = findOutputNode(tg, result.expected().type());
        if (outputTNode == null) return null;

        State state = new State(tg, mandates, rootInput, budget);
        CompGraphNode terminal = state.createNode(outputTNode);

        if (!solve(state, terminal, 0, result.expected(), 0, new ArrayList<>())) {
            return null;
        }
        state.completed.add(terminal);

        for (Mandate m : mandates.mandates()) {
            if (m.isResult()) continue;
            if (alreadyProduced(state, m.expected())) continue;
            if (!produceMandate(state, m.expected())) {
                return null;
            }
        }

        ComputationGraph cg = new ComputationGraph(state.nodes, terminal);
        for (RootBinding rb : state.rootBindings) {
            cg.bindRoot(rb.node(), rb.slot(), rb.value());
        }
        return new CarvingResult(
                cg,
                List.copyOf(state.tracedEdges),
                Map.copyOf(state.simulatedValues));
    }

    private boolean alreadyProduced(State state, Value target) {
        for (CompGraphNode n : state.completed) {
            Value sim = state.simulatedValues.get(n);
            if (sim != null && sim.type() == target.type()
                    && ValueDistance.matches(sim, target, 1e-6)) return true;
        }
        return false;
    }

    private boolean produceMandate(State state, Value target) {
        List<TransformationNode> candidates = new ArrayList<>(state.tg.nodesProducing(target.type()));
        Collections.shuffle(candidates, rng);
        if (rng.nextDouble() >= epsilon) {
            candidates.sort(Comparator.comparingDouble((TransformationNode tn) -> {
                double avg = 0.0;
                int n = 0;
                for (TransformationEdge e : state.tg.outgoing(tn)) {
                    if (e.stats().isPruned()) continue;
                    avg += e.stats().score();
                    n++;
                }
                return n == 0 ? -1 : avg / n;
            }).reversed());
        }

        for (TransformationNode cand : candidates) {
            if (cand.primitive() instanceof sibarum.strnn.primitive.Terminal) continue;
            List<Value> inputs = inferInputs(cand.primitive(), target, state);
            if (inputs == null) continue;

            int snapNodes = state.nodes.size();
            int snapEdges = state.tracedEdges.size();
            int snapRoots = state.rootBindings.size();
            int snapBudget = state.budget;
            List<Map.Entry<CompGraphNode, Value>> snapSimList = new ArrayList<>(state.simulatedValues.entrySet());
            Set<CompGraphNode> snapCompleted = new HashSet<>(state.completed);

            CompGraphNode newNode = state.createNode(cand);
            state.simulatedValues.put(newNode, target);

            List<PathEntry> path = new ArrayList<>();
            path.add(new PathEntry(target, cand.primitive().getClass()));
            boolean allOk = true;
            for (int i = 0; i < cand.inputTypes().size(); i++) {
                if (!solve(state, newNode, i, inputs.get(i), 1, path)) {
                    allOk = false;
                    break;
                }
            }

            if (allOk) {
                state.completed.add(newNode);
                return true;
            }

            truncate(state.nodes, snapNodes);
            truncate(state.tracedEdges, snapEdges);
            truncate(state.rootBindings, snapRoots);
            state.simulatedValues.clear();
            for (Map.Entry<CompGraphNode, Value> e : snapSimList) {
                state.simulatedValues.put(e.getKey(), e.getValue());
            }
            state.completed.clear();
            state.completed.addAll(snapCompleted);
            state.budget = snapBudget;
        }
        return false;
    }

    private TransformationNode findOutputNode(TransformationGraph tg, ValueType resultType) {
        for (TransformationNode n : tg.nodes()) {
            Primitive p = n.primitive();
            if (resultType == ValueType.PARSE_TREE
                    && p instanceof sibarum.strnn.primitive.TreeOutputPrimitive) return n;
            if (resultType != ValueType.PARSE_TREE && p instanceof OutputPrimitive) return n;
        }
        return null;
    }

    private boolean solve(State state, CompGraphNode parent, int slot, Value target, int depth, List<PathEntry> targetPath) {
        if (state.budget <= 0 || depth > MAX_DEPTH) return false;
        state.budget--;

        ValueType wantType = parent.tNode().inputTypes().get(slot);
        if (target.type() != wantType) return false;

        if (state.rootInput.type() == wantType
                && ValueDistance.matches(state.rootInput, target, 0.0)) {
            state.rootBindings.add(new RootBinding(parent, slot, state.rootInput));
            return true;
        }

        for (CompGraphNode existing : state.completed) {
            if (existing == parent) continue;
            Value sim = state.simulatedValues.get(existing);
            if (sim != null && sim.type() == wantType
                    && ValueDistance.matches(sim, target, 1e-6)) {
                TransformationEdge edge = state.tg.edge(existing.tNode(), parent.tNode());
                if (edge == null || edge.stats().isPruned()) continue;
                parent.wire(slot, new SlotSource(existing, edge));
                state.tracedEdges.add(edge);
                return true;
            }
        }

        List<TransformationNode> candidates = rankedCandidates(state, parent.tNode(), wantType, target);
        for (TransformationNode cand : candidates) {
            if (cand.primitive() instanceof sibarum.strnn.primitive.Terminal) continue;
            TransformationEdge edge = state.tg.edge(cand, parent.tNode());
            if (edge == null || edge.stats().isPruned()) continue;

            if (pathContains(targetPath, target, cand.primitive().getClass())) continue;

            List<Value> inputs = inferInputs(cand.primitive(), target, state);
            if (inputs == null) continue;

            int snapNodes = state.nodes.size();
            int snapEdges = state.tracedEdges.size();
            int snapRoots = state.rootBindings.size();
            int snapBudget = state.budget;
            List<Map.Entry<CompGraphNode, Value>> snapSimList = new ArrayList<>(state.simulatedValues.entrySet());
            Set<CompGraphNode> snapCompleted = new HashSet<>(state.completed);

            CompGraphNode newNode = state.createNode(cand);
            parent.wire(slot, new SlotSource(newNode, edge));
            state.tracedEdges.add(edge);
            state.simulatedValues.put(newNode, target);

            targetPath.add(new PathEntry(target, cand.primitive().getClass()));
            boolean allOk = true;
            for (int i = 0; i < cand.inputTypes().size(); i++) {
                if (!solve(state, newNode, i, inputs.get(i), depth + 1, targetPath)) {
                    allOk = false;
                    break;
                }
            }
            targetPath.removeLast();

            if (allOk) {
                state.completed.add(newNode);
                return true;
            }

            unwire(parent, slot);
            truncate(state.nodes, snapNodes);
            truncate(state.tracedEdges, snapEdges);
            truncate(state.rootBindings, snapRoots);
            state.simulatedValues.clear();
            for (Map.Entry<CompGraphNode, Value> e : snapSimList) {
                state.simulatedValues.put(e.getKey(), e.getValue());
            }
            state.completed.clear();
            state.completed.addAll(snapCompleted);
            state.budget = snapBudget;
        }
        return false;
    }

    private static boolean pathContains(List<PathEntry> path, Value target, Class<?> primClass) {
        for (PathEntry pe : path) {
            if (pe.primClass.equals(primClass)
                    && pe.target.type() == target.type()
                    && ValueDistance.matches(pe.target, target, 1e-9)) {
                return true;
            }
        }
        return false;
    }

    private record PathEntry(Value target, Class<?> primClass) {
    }

    private static void unwire(CompGraphNode parent, int slot) {
        parent.clearSlot(slot);
    }

    private static <T> void truncate(List<T> list, int size) {
        while (list.size() > size) list.removeLast();
    }

    private List<TransformationNode> rankedCandidates(
            State state, TransformationNode parent, ValueType wantType, Value target) {
        List<TransformationNode> pool = new ArrayList<>(state.tg.nodesProducing(wantType));
        Collections.shuffle(pool, rng);
        if (rng.nextDouble() >= epsilon) {
            pool.sort(Comparator.comparingDouble((TransformationNode tn) -> {
                TransformationEdge e = state.tg.edge(tn, parent);
                if (e == null || e.stats().isPruned()) return Double.NEGATIVE_INFINITY;
                return e.stats().score();
            }).reversed());
        }

        List<TransformationNode> hits = new ArrayList<>();
        List<TransformationNode> rest = new ArrayList<>();
        for (TransformationNode tn : pool) {
            if (candidateProducesMandateValue(state, tn, target)) {
                hits.add(tn);
            } else {
                rest.add(tn);
            }
        }
        hits.addAll(rest);
        return hits;
    }

    private boolean candidateProducesMandateValue(State state, TransformationNode cand, Value target) {
        List<Value> inputs = inferInputs(cand.primitive(), target, state);
        if (inputs == null) return false;
        for (Value v : inputs) {
            for (Mandate m : state.mandates.mandates()) {
                if (m.isResult()) continue;
                if (m.expected().type() == v.type()
                        && ValueDistance.matches(m.expected(), v, 1e-6)) {
                    return true;
                }
            }
        }
        return false;
    }

    private List<Value> inferInputs(Primitive prim, Value target, State state) {
        if (prim instanceof OutputPrimitive) {
            return List.of(target);
        }
        if (prim instanceof ParseNumber && target instanceof NumberValue(double n)) {
            String s = isInteger(n) ? Integer.toString((int) Math.round(n)) : Double.toString(n);
            return List.of(new StringValue(s));
        }
        if (prim instanceof NumberToMatrix && target instanceof MatrixValue mv && mv.dim() == 1) {
            return List.of(new NumberValue(mv.data()[0] * NumberToMatrix.SCALE));
        }
        if (prim instanceof MatrixToNumber && target instanceof NumberValue(double n)) {
            return List.of(new MatrixValue(new double[]{n / NumberToMatrix.SCALE}));
        }
        if (prim instanceof ComposeMatrices && target instanceof MatrixValue mv && mv.dim() == 2) {
            return List.of(
                    new MatrixValue(new double[]{mv.data()[0]}),
                    new MatrixValue(new double[]{mv.data()[1]}));
        }
        if (prim instanceof SplitStringAt sp && target instanceof TokenListValue(List<String> tokens)) {
            return List.of(new StringValue(String.join(String.valueOf(sp.delim()), tokens)));
        }
        if (prim instanceof TokenAt ta && target instanceof StringValue(String s)) {
            for (Mandate m : state.mandates.mandates()) {
                if (m.expected() instanceof TokenListValue tlv
                        && ta.index() < tlv.tokens().size()
                        && tlv.tokens().get(ta.index()).equals(s)) {
                    return List.of(tlv);
                }
            }
            return null;
        }
        if (prim instanceof LearnedArithmetic la && target instanceof MatrixValue mv && mv.dim() == 1
                && !(prim instanceof EvaluateBinaryOp)) {
            double targetScaled = mv.data()[0];
            double targetReal = targetScaled * NumberToMatrix.SCALE;
            List<NumberValue> pool = numericAnchors(state);
            for (NumberValue a : pool) {
                for (NumberValue b : pool) {
                    boolean ok = switch (la.role()) {
                        case ADD -> Math.abs(a.n() + b.n() - targetReal) < 1e-6;
                        case MUL -> Math.abs(a.n() * b.n() - targetReal) < 1e-6;
                    };
                    if (ok) {
                        return List.of(new MatrixValue(new double[]{
                                a.n() / NumberToMatrix.SCALE,
                                b.n() / NumberToMatrix.SCALE}));
                    }
                }
            }
            return null;
        }
        if (prim instanceof EvaluateBinaryOp eval && target instanceof ParseTreeValue.Literal lit) {
            // Inverter: target Lit(v) <- BinaryOp(op, Lit(a), Lit(b)) where a op b ≈ v.
            double targetReal = lit.value();
            List<NumberValue> pool = numericAnchors(state);
            for (NumberValue a : pool) {
                for (NumberValue b : pool) {
                    boolean ok = switch (eval.role()) {
                        case ADD -> Math.abs(a.n() + b.n() - targetReal) < 1e-6;
                        case MUL -> Math.abs(a.n() * b.n() - targetReal) < 1e-6;
                    };
                    if (ok) {
                        sibarum.strnn.value.Operator op = (eval.role() == sibarum.strnn.primitive.MlpRole.ADD)
                                ? sibarum.strnn.value.Operator.ADD
                                : sibarum.strnn.value.Operator.MUL;
                        return List.of(new ParseTreeValue.BinaryOp(
                                op,
                                new ParseTreeValue.Literal(a.n()),
                                new ParseTreeValue.Literal(b.n())));
                    }
                }
            }
            return null;
        }
        if (prim instanceof RewriteRulePrimitive rule && target instanceof ParseTreeValue pt) {
            return rule.inferInput(pt).map(t -> List.<Value>of(t)).orElse(null);
        }
        if (prim instanceof TreeOutputPrimitive && target instanceof ParseTreeValue) {
            return List.of(target);
        }
        if (prim instanceof ParseExpression && target instanceof ParseTreeValue pt) {
            // Only invertible if the available root input parses to the target tree.
            if (state.rootInput instanceof StringValue) {
                try {
                    Value parsed = prim.apply(List.of(state.rootInput));
                    if (parsed.equals(pt)) return List.of(state.rootInput);
                } catch (Exception ignored) {
                }
            }
            return null;
        }
        return null;
    }

    private static List<NumberValue> numericAnchors(State state) {
        List<NumberValue> pool = new ArrayList<>();
        for (Mandate m : state.mandates.mandates()) {
            collectNumbers(m.expected(), pool);
        }
        return pool;
    }

    private static void collectNumbers(Value v, List<NumberValue> out) {
        if (v instanceof NumberValue nv) {
            out.add(nv);
        } else if (v instanceof TokenListValue(List<String> tokens)) {
            for (String t : tokens) {
                try {
                    out.add(new NumberValue(Double.parseDouble(t)));
                } catch (NumberFormatException ignored) {
                }
            }
        } else if (v instanceof ParseTreeValue pt) {
            collectNumbersFromTree(pt, out);
        }
    }

    private static void collectNumbersFromTree(ParseTreeValue t, List<NumberValue> out) {
        switch (t) {
            case ParseTreeValue.Literal lit -> out.add(new NumberValue(lit.value()));
            case ParseTreeValue.BinaryOp bo -> {
                collectNumbersFromTree(bo.left(), out);
                collectNumbersFromTree(bo.right(), out);
            }
            case ParseTreeValue.Variable ignored -> {
            }
            case ParseTreeValue.Omega ignored -> {
            }
        }
    }

    private static boolean isInteger(double d) {
        return Math.abs(d - Math.round(d)) < 1e-9;
    }

    private static final class State {
        final TransformationGraph tg;
        final MandateSet mandates;
        final Value rootInput;
        final List<CompGraphNode> nodes = new ArrayList<>();
        final List<TransformationEdge> tracedEdges = new ArrayList<>();
        final List<RootBinding> rootBindings = new ArrayList<>();
        final Map<CompGraphNode, Value> simulatedValues = new HashMap<>();
        final Set<CompGraphNode> completed = new HashSet<>();
        int budget;
        int uid = 0;

        State(TransformationGraph tg, MandateSet mandates, Value rootInput, int budget) {
            this.tg = tg;
            this.mandates = mandates;
            this.rootInput = rootInput;
            this.budget = budget;
        }

        CompGraphNode createNode(TransformationNode t) {
            CompGraphNode n = new CompGraphNode("c" + (uid++) + "_" + t.id(), t);
            nodes.add(n);
            return n;
        }
    }

    private record RootBinding(CompGraphNode node, int slot, Value value) {
    }
}
