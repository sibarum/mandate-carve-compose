package sibarum.elden.graph;

import sibarum.elden.annotation.EntityType;

import java.util.Optional;
import java.util.Set;

/**
 * A unique entity in the cross-item graph. Type is Optional because an entity
 * may be declared only in relations (never appearing as a span in any prose),
 * in which case we have no surface form to attach a type to.
 */
public record EntityNode(
        String id,
        Optional<EntityType> type,
        Set<String> surfaceForms,
        int spanOccurrences,
        int relationOccurrences
) {}
