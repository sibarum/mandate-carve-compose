package sibarum.strnn.carving;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.cache.CachedNetworkPrimitive;
import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.LearnedArithmetic;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.Terminal;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.primitive.Trainable;
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
        precomputeForwardAnchors(state);
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
                Map.copyOf(state.simulatedValues),
                List.copyOf(state.rootBindings));
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
            List<Value> inputs = inferInputs(cand, target, state);
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
            if (n.primitive() instanceof Terminal && n.outputType() == resultType) return n;
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

            List<Value> inputs = inferInputs(cand, target, state);
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
        List<Value> inputs = inferInputs(cand, target, state);
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

    private List<Value> inferInputs(TransformationNode candNode, Value target, State state) {
        Primitive prim = candNode.primitive();
        if (prim instanceof Terminal) {
            return List.of(target);
        }
        if (prim instanceof LookupSymbol ls && target instanceof StringValue(String s)) {
            // LookupSymbol's inversion is exact: the matrix that lookup-to-s requires
            // is the symbol's canonical embedding.
            return List.of(new MatrixValue(ls.table().embed(s)));
        }
        if (prim instanceof EmbedSymbol es && target instanceof MatrixValue mv) {
            // EmbedSymbol's inversion: which symbol's embedding is the target?
            // Use cosine nearest. If no symbol resolves, this path fails.
            return es.table().nearest(mv.data())
                    .map(s -> List.<Value>of(new StringValue(s)))
                    .orElse(null);
        }
        if (prim instanceof VectorTransform vt && target instanceof MatrixValue) {
            // Trainable: input doesn't need to be a specific value (the bridge
            // learns to map whatever comes in). The carver needs *some* concrete
            // matrix value of the right dim so solve() can recurse on it and
            // terminate at the root. Pick the anchor produced by this node's
            // TG-upstream candidate with the best edge stat — that's the head
            // -aware choice when multiple parallel chains exist.
            Value anchor = pickBridgeAnchor(state, candNode, vt.inDim());
            if (anchor != null) return List.of(anchor);
            return null;
        }
        if (prim instanceof CachedNetworkPrimitive cnp) {
            // Deterministic cached subgraph. Find an input value of the
            // right type such that the cached network's forward evaluation
            // produces the target. The reachable-values set
            // (populated via BFS in the forward pre-pass) lets the carver
            // chain N cached networks: a value reachable only after N-1
            // applications of upstream primitives is still in the set and
            // is a valid input candidate for this primitive.
            ValueType wantType = cnp.item().inputType();
            if (target.type() != cnp.item().outputType()) return null;
            Set<Value> candidates = state.reachableValuesByType.getOrDefault(
                    wantType, Collections.emptySet());
            for (Value v : candidates) {
                Value out = safeApply(cnp, v);
                if (out != null && ValueDistance.matches(out, target, 1e-6)) {
                    return List.of(v);
                }
            }
            return null;
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
        }
    }

    private static boolean isInteger(double d) {
        return Math.abs(d - Math.round(d)) < 1e-9;
    }

    /**
     * Walk forward from the rootInput. Two structures are populated:
     *
     *   forwardAnchorsByNode: one representative output per TransformationNode.
     *   This drives trainable inversion (the bridge picks its upstream
     *   anchor from its TG-incoming edge).
     *
     *   reachableValuesByType: BFS over all values reachable from rootInput
     *   by chaining any number of non-Terminal primitives. This is what
     *   lets the carver chain cached networks N levels deep — for an
     *   inversion of "primitive P needs an input that produces target",
     *   we try every reachable value of the right type, not just the
     *   root or one per-node anchor.
     *
     * Trainables are included in both: their current forward function is
     * well-defined.
     */
    private void precomputeForwardAnchors(State state) {
        state.rootAnchor = state.rootInput;
        state.reachableValuesByType
                .computeIfAbsent(state.rootInput.type(), k -> new java.util.LinkedHashSet<>())
                .add(state.rootInput);
        // BFS reachability is only useful when the substrate contains
        // primitives that compose deterministically over the input space
        // (e.g., CachedNetworkPrimitive — Key-Network). Skipping it for
        // substrates without such primitives keeps the pre-pass fast and
        // bounded for the common KV-cache / trainable-bridge case.
        boolean bfsEnabled = false;
        for (TransformationNode tn : state.tg.nodes()) {
            if (tn.primitive() instanceof CachedNetworkPrimitive) {
                bfsEnabled = true;
                break;
            }
        }
        boolean changed = true;
        int guard = 0;
        while (changed && guard++ < 32) {
            changed = false;
            for (TransformationNode tn : state.tg.nodes()) {
                Primitive p = tn.primitive();
                if (p instanceof Terminal) continue;

                // Build the per-node anchor (one value per node, used by
                // trainable inversion) if not already done.
                if (!state.forwardAnchorsByNode.containsKey(tn)) {
                    List<Value> ins = new ArrayList<>();
                    boolean ok = true;
                    for (ValueType t : tn.inputTypes()) {
                        Value v = anchorByType(state, t);
                        if (v == null) {
                            ok = false;
                            break;
                        }
                        ins.add(v);
                    }
                    if (ok) {
                        try {
                            Value out = p.apply(ins);
                            state.forwardAnchorsByNode.put(tn, out);
                            state.reachableValuesByType
                                    .computeIfAbsent(out.type(), k -> new java.util.LinkedHashSet<>())
                                    .add(out);
                            changed = true;
                        } catch (RuntimeException ignored) {
                        }
                    }
                }

                // BFS step: extend reachableValuesByType by trying every
                // already-reachable input combination through this primitive.
                // Restricted to single-input, non-Trainable primitives:
                //   - Single-input: most cases (cached networks, lookups,
                //     embed-style); multi-input would need a cross product
                //     and is not needed for the current demos.
                //   - Non-Trainable: a Trainable's forward function can
                //     produce arbitrary continuous outputs that aren't
                //     guaranteed to be near a vocabulary element. BFSing
                //     through one would generate unbounded distinct matrix
                //     values. Trainables still get their per-node anchor
                //     above for solve()'s recursion to terminate on.
                if (bfsEnabled && tn.inputTypes().size() == 1 && !(p instanceof Trainable)) {
                    Set<Value> options = state.reachableValuesByType
                            .getOrDefault(tn.inputTypes().getFirst(), Collections.emptySet());
                    for (Value v : new ArrayList<>(options)) {
                        try {
                            Value out = p.apply(List.of(v));
                            Set<Value> outSet = state.reachableValuesByType
                                    .computeIfAbsent(out.type(), k -> new java.util.LinkedHashSet<>());
                            if (outSet.size() >= 1024) continue;
                            if (outSet.add(out)) {
                                changed = true;
                            }
                        } catch (RuntimeException ignored) {
                        }
                    }
                }
            }
        }
    }

    /**
     * Forward-apply a CachedNetworkPrimitive against one input value,
     * catching any runtime failure (dim mismatch, type mismatch from an
     * inner primitive, etc.). Returns null on failure.
     */
    private static Value safeApply(CachedNetworkPrimitive cnp, Value input) {
        try {
            return cnp.apply(List.of(input));
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    /**
     * Find any anchor value of the given type. Used for joining-primitive
     * forward propagation when we don't yet know which specific source feeds
     * which slot. Returns the root input if it matches; otherwise the first
     * matching per-node anchor.
     */
    private static Value anchorByType(State state, ValueType t) {
        if (state.rootAnchor != null && state.rootAnchor.type() == t) return state.rootAnchor;
        for (Value v : state.forwardAnchorsByNode.values()) {
            if (v.type() == t) return v;
        }
        return null;
    }

    /**
     * For a trainable bridge candidate (a {@link VectorTransform}), pick the
     * forward anchor of an upstream TG-source. Prefers upstreams ranked by
     * edge-stat score on the {@code (upstream → bridge)} edge — so when the
     * substrate has multiple parallel chains, the bridge gets paired with the
     * head whose edge has been reinforced.
     *
     * Falls back to any matrix anchor of the right dimensionality.
     */
    private Value pickBridgeAnchor(State state, TransformationNode bridgeNode, int wantDim) {
        // Rank incoming TG edges by stat score (highest first).
        List<TransformationEdge> incoming = new ArrayList<>(state.tg.incoming(bridgeNode));
        incoming.sort(Comparator.comparingDouble((TransformationEdge e) ->
                e.stats().isPruned() ? Double.NEGATIVE_INFINITY : e.stats().score()).reversed());
        for (TransformationEdge e : incoming) {
            if (e.stats().isPruned()) continue;
            Value anchor = state.forwardAnchorsByNode.get(e.from());
            if (anchor instanceof MatrixValue mv && mv.dim() == wantDim) return anchor;
            // root-fed bridges: source has no anchor entry but root is the source
            if (anchor == null && state.rootAnchor instanceof MatrixValue rm && rm.dim() == wantDim
                    && state.rootInput.type() == e.from().outputType()) {
                return rm;
            }
        }
        // Fallback: any matrix anchor of the right dim. Lets the carver still
        // recover if the TG topology doesn't point at a viable upstream.
        for (Value v : state.forwardAnchorsByNode.values()) {
            if (v instanceof MatrixValue mv && mv.dim() == wantDim) return v;
        }
        if (state.rootAnchor instanceof MatrixValue rm && rm.dim() == wantDim) return rm;
        return null;
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
        // Per-source forward anchors. Keyed by TransformationNode so multi-head
        // substrates keep parallel-chain anchors separate. The root input is
        // stored separately under rootAnchor (no TransformationNode for it).
        final Map<TransformationNode, Value> forwardAnchorsByNode = new HashMap<>();
        // All values reachable by chaining non-Terminal primitives forward
        // from rootInput, indexed by type. Drives chained-composition
        // inversion for deterministic primitives (most notably
        // CachedNetworkPrimitive).
        final Map<ValueType, Set<Value>> reachableValuesByType = new HashMap<>();
        Value rootAnchor;
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

}
