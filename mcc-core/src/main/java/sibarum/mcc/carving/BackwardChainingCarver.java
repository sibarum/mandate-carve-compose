package sibarum.mcc.carving;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationEdge;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationNode;
import sibarum.mcc.mandate.Mandate;
import sibarum.mcc.mandate.MandateSet;
import sibarum.mcc.op.Terminal;
import sibarum.mcc.primitive.Inversion;
import sibarum.mcc.primitive.InversionContext;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueDistance;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

/**
 * Backward-chaining carver. Starts at the result mandate's value
 * attached to a Terminal node, then for each unsolved slot tries
 * (in order): bind to the root input if it matches, wire to an
 * already-placed node whose simulated value matches, or invert the
 * candidate primitive (via {@link Inversion}) to discover what
 * upstream value would produce the target and recurse on it.
 *
 * <p>Per-primitive inversion lives on the primitive itself
 * ({@link sibarum.mcc.primitive.Primitive#inversion}); the carver
 * contains no primitive-specific branches.
 *
 * <p>Candidates are ranked by their edge-stats score on the
 * {@code (candidate → parent)} substrate edge. A non-zero
 * {@code epsilon} adds random-order exploration so alternative
 * paths accumulate samples.
 */
public final class BackwardChainingCarver {

    private static final int DEFAULT_BUDGET = 200;
    private static final int MAX_DEPTH = 16;
    private static final double MATCH_TOL = 1e-6;

    private final Random rng;
    private final int budget;
    private final double epsilon;

    public BackwardChainingCarver(long seed) {
        this(seed, DEFAULT_BUDGET, 0.0);
    }

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
        TransformationNode terminalNode = findTerminalNode(tg, result.expected().type());
        if (terminalNode == null) return null;

        State state = new State(tg, mandates, rootInput, budget);
        precomputeReachableValues(state);

        CompGraphNode terminal = state.createNode(terminalNode);
        if (!solve(state, terminal, 0, result.expected(), 0)) return null;
        state.completed.add(terminal);
        state.simulatedValues.put(terminal, result.expected());

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

    private boolean solve(State state, CompGraphNode parent, int slot, Value target, int depth) {
        if (state.budget <= 0 || depth > MAX_DEPTH) return false;
        state.budget--;
        ValueType wantType = parent.tNode().inputTypes().get(slot);
        if (target.type() != wantType) return false;

        // Option (a): bind to root input.
        if (state.rootInput.type() == wantType
                && ValueDistance.matches(state.rootInput, target, MATCH_TOL)) {
            state.rootBindings.add(new RootBinding(parent, slot, state.rootInput));
            return true;
        }

        // Option (b): wire to an already-placed node whose simulated value matches.
        for (CompGraphNode existing : state.completed) {
            if (existing == parent) continue;
            Value sim = state.simulatedValues.get(existing);
            if (sim == null || sim.type() != wantType) continue;
            if (!ValueDistance.matches(sim, target, MATCH_TOL)) continue;
            TransformationEdge edge = state.tg.edge(existing.tNode(), parent.tNode());
            if (edge == null || edge.stats().isPruned()) continue;
            parent.wire(slot, new SlotSource(existing, edge));
            state.tracedEdges.add(edge);
            return true;
        }

        // Option (c): create a new node by inverting a candidate primitive.
        List<TransformationNode> candidates = rankedCandidates(state, parent.tNode(), wantType);
        for (TransformationNode cand : candidates) {
            if (cand.primitive() instanceof Terminal) continue;
            TransformationEdge edge = state.tg.edge(cand, parent.tNode());
            if (edge == null || edge.stats().isPruned()) continue;

            Inversion inv = cand.primitive().inversion();
            List<Value> inputs = inv.invert(target, state);
            if (inputs == null) continue;
            if (inputs.size() != cand.inputTypes().size()) continue;

            // Snapshot for backtracking.
            int snapNodes = state.nodes.size();
            int snapEdges = state.tracedEdges.size();
            int snapRoots = state.rootBindings.size();
            int snapBudget = state.budget;
            Map<CompGraphNode, Value> snapSim = new HashMap<>(state.simulatedValues);
            Set<CompGraphNode> snapCompleted = new HashSet<>(state.completed);

            CompGraphNode newNode = state.createNode(cand);
            parent.wire(slot, new SlotSource(newNode, edge));
            state.tracedEdges.add(edge);
            state.simulatedValues.put(newNode, target);

            boolean allOk = true;
            for (int i = 0; i < cand.inputTypes().size(); i++) {
                if (!solve(state, newNode, i, inputs.get(i), depth + 1)) {
                    allOk = false;
                    break;
                }
            }

            if (allOk) {
                state.completed.add(newNode);
                return true;
            }

            // Backtrack.
            parent.clearSlot(slot);
            truncate(state.nodes, snapNodes);
            truncate(state.tracedEdges, snapEdges);
            truncate(state.rootBindings, snapRoots);
            state.simulatedValues.clear();
            state.simulatedValues.putAll(snapSim);
            state.completed.clear();
            state.completed.addAll(snapCompleted);
            state.budget = snapBudget;
        }
        return false;
    }

    private List<TransformationNode> rankedCandidates(State state, TransformationNode parent, ValueType wantType) {
        List<TransformationNode> pool = new ArrayList<>(state.tg.nodesProducing(wantType));
        Collections.shuffle(pool, rng);
        if (rng.nextDouble() >= epsilon) {
            pool.sort(Comparator.comparingDouble((TransformationNode tn) -> {
                TransformationEdge e = state.tg.edge(tn, parent);
                if (e == null || e.stats().isPruned()) return Double.NEGATIVE_INFINITY;
                return e.stats().score();
            }).reversed());
        }
        return pool;
    }

    private static TransformationNode findTerminalNode(TransformationGraph tg, ValueType resultType) {
        for (TransformationNode n : tg.nodes()) {
            if (n.primitive() instanceof Terminal && n.outputType() == resultType) return n;
        }
        return null;
    }

    private static <T> void truncate(List<T> list, int size) {
        while (list.size() > size) list.removeLast();
    }

    /**
     * Forward-BFS over the substrate from the root input. Populates
     * {@code reachableValuesByType} with every value reachable by
     * chaining non-Terminal, non-Trainable primitives. The result is
     * the value pool that {@link InversionContext#reachableValuesOfType}
     * and {@link InversionContext#anchorByMatrixDim} draw from.
     *
     * <p>Trainable primitives are skipped — their forward output isn't
     * a stable vocabulary element, and BFSing through them would
     * explode the value set.
     */
    private static void precomputeReachableValues(State state) {
        state.reachableValuesByType
                .computeIfAbsent(state.rootInput.type(), k -> new LinkedHashSet<>())
                .add(state.rootInput);
        boolean changed = true;
        int guard = 0;
        while (changed && guard++ < 32) {
            changed = false;
            for (TransformationNode tn : state.tg.nodes()) {
                if (tn.primitive() instanceof Terminal) continue;
                if (tn.primitive() instanceof sibarum.mcc.primitive.Trainable) continue;
                if (tn.inputTypes().size() != 1) continue;  // single-input BFS for now
                ValueType inType = tn.inputTypes().getFirst();
                Set<Value> options = state.reachableValuesByType
                        .getOrDefault(inType, Collections.emptySet());
                for (Value v : new ArrayList<>(options)) {
                    try {
                        Value out = tn.primitive().apply(List.of(v));
                        Set<Value> outSet = state.reachableValuesByType
                                .computeIfAbsent(out.type(), k -> new LinkedHashSet<>());
                        if (outSet.add(out)) changed = true;
                    } catch (RuntimeException ignored) {
                        // Primitive rejected this input; that's fine.
                    }
                }
            }
        }
    }

    /** Carver internal state; implements {@link InversionContext}. */
    private static final class State implements InversionContext {
        final TransformationGraph tg;
        final MandateSet mandates;
        final Value rootInput;
        final List<CompGraphNode> nodes = new ArrayList<>();
        final List<TransformationEdge> tracedEdges = new ArrayList<>();
        final List<RootBinding> rootBindings = new ArrayList<>();
        final Map<CompGraphNode, Value> simulatedValues = new HashMap<>();
        final Set<CompGraphNode> completed = new HashSet<>();
        final Map<ValueType, Set<Value>> reachableValuesByType = new HashMap<>();
        final Random rng = new Random(0L);
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

        @Override
        public Value rootInput() {
            return rootInput;
        }

        @Override
        public List<Value> reachableValuesOfType(ValueType type) {
            Set<Value> s = reachableValuesByType.get(type);
            return s == null ? List.of() : List.copyOf(s);
        }

        @Override
        public List<Value> mandateValues() {
            List<Value> out = new ArrayList<>(mandates.size());
            for (Mandate m : mandates.mandates()) out.add(m.expected());
            return out;
        }

        @Override
        public Optional<MatrixValue> anchorByMatrixDim(int dim) {
            // Prefer reachable values of MATRIX type; fall back to mandate
            // values; fall back to the root input.
            Set<Value> reach = reachableValuesByType.get(ValueType.MATRIX);
            if (reach != null) {
                for (Value v : reach) {
                    if (v instanceof MatrixValue mv && mv.dim() == dim) return Optional.of(mv);
                }
            }
            for (Mandate m : mandates.mandates()) {
                if (m.expected() instanceof MatrixValue mv && mv.dim() == dim) {
                    return Optional.of(mv);
                }
            }
            if (rootInput instanceof MatrixValue mv && mv.dim() == dim) {
                return Optional.of(mv);
            }
            return Optional.empty();
        }

        @Override
        public Random rng() {
            return rng;
        }
    }
}
