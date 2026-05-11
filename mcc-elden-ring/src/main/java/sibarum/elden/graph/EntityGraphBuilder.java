package sibarum.elden.graph;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.EntitySpan;
import sibarum.elden.annotation.EntityType;
import sibarum.elden.annotation.Relation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class EntityGraphBuilder {

    private EntityGraphBuilder() {}

    public static EntityGraph build(List<AnnotatedItem> items) {
        return build(items, Map.of());
    }

    public static EntityGraph build(List<AnnotatedItem> items, Map<String, EntityType> implicit) {
        Map<String, EntityType> typeById = new LinkedHashMap<>();
        Map<String, Set<String>> surfacesById = new LinkedHashMap<>();
        Map<String, Integer> spanCount = new HashMap<>();
        Map<String, Integer> relationDegree = new HashMap<>();
        List<Relation> allRelations = new ArrayList<>();

        for (AnnotatedItem item : items) {
            for (EntitySpan span : item.spans()) {
                EntityType prior = typeById.get(span.entityId());
                if (prior != null && prior != span.type()) {
                    throw new IllegalStateException(
                            "entity '" + span.entityId() + "' has conflicting types: "
                                    + prior + " (earlier) vs " + span.type() + " (in item " + item.itemId() + ")");
                }
                typeById.put(span.entityId(), span.type());
                surfacesById.computeIfAbsent(span.entityId(), k -> new LinkedHashSet<>()).add(span.surface());
                spanCount.merge(span.entityId(), 1, Integer::sum);
            }
            for (Relation r : item.relations()) {
                surfacesById.putIfAbsent(r.subjectId(), new LinkedHashSet<>());
                surfacesById.putIfAbsent(r.objectId(), new LinkedHashSet<>());
                relationDegree.merge(r.subjectId(), 1, Integer::sum);
                relationDegree.merge(r.objectId(), 1, Integer::sum);
                allRelations.add(r);
            }
        }

        for (Map.Entry<String, EntityType> e : implicit.entrySet()) {
            EntityType spanType = typeById.get(e.getKey());
            if (spanType != null && spanType != e.getValue()) {
                throw new IllegalStateException(
                        "implicit type for '" + e.getKey() + "' conflicts with span type: "
                                + e.getValue() + " (implicit) vs " + spanType + " (span)");
            }
            typeById.putIfAbsent(e.getKey(), e.getValue());
            surfacesById.putIfAbsent(e.getKey(), new LinkedHashSet<>());
        }

        Map<String, EntityNode> nodes = new LinkedHashMap<>();
        for (String id : surfacesById.keySet()) {
            nodes.put(id, new EntityNode(
                    id,
                    Optional.ofNullable(typeById.get(id)),
                    Set.copyOf(surfacesById.get(id)),
                    spanCount.getOrDefault(id, 0),
                    relationDegree.getOrDefault(id, 0)
            ));
        }

        return new EntityGraph(Map.copyOf(nodes), List.copyOf(allRelations));
    }
}
