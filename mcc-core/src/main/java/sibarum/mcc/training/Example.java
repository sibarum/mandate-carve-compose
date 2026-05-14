package sibarum.mcc.training;

import sibarum.mcc.mandate.MandateSet;
import sibarum.mcc.value.Value;

import java.util.Map;
import java.util.Objects;

/**
 * One training instance for a {@link sibarum.mcc.graph.ComputationGraph}.
 *
 * <p>{@code inputs} is a name → Value map; the {@link GraphTrainer}'s
 * configured binder converts each name to the right
 * {@code (CompGraphNode, slotIndex)} root binding before execution.
 *
 * <p>{@code target} is the expected output value at the graph's
 * terminal node; trainers use it to drive {@code backward}.
 *
 * <p>{@code mandates} is an optional declarative spec checked by
 * {@link sibarum.mcc.mandate.MandateVerifier}. Pass an empty
 * {@code MandateSet} if no structural constraints are required.
 */
public record Example(String label, Map<String, Value> inputs, Value target, MandateSet mandates) {

    public Example {
        Objects.requireNonNull(label);
        Objects.requireNonNull(inputs);
        Objects.requireNonNull(target);
        Objects.requireNonNull(mandates);
        inputs = Map.copyOf(inputs);
    }

    public Example(String label, Map<String, Value> inputs, Value target) {
        this(label, inputs, target, new MandateSet(java.util.List.of()));
    }
}
