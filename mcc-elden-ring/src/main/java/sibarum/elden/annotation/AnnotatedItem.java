package sibarum.elden.annotation;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public record AnnotatedItem(
        String itemId,
        String rawText,
        List<EntitySpan> spans,
        List<Relation> relations
) {

    public Set<String> entityIdsInText() {
        Set<String> ids = new LinkedHashSet<>();
        for (EntitySpan span : spans) ids.add(span.entityId());
        return ids;
    }

    public Set<String> entityIdsInRelations() {
        Set<String> ids = new LinkedHashSet<>();
        for (Relation r : relations) {
            ids.add(r.subjectId());
            ids.add(r.objectId());
        }
        return ids;
    }

    public Set<String> entitiesReferencedButNotInText() {
        Set<String> inText = entityIdsInText();
        Set<String> missing = new LinkedHashSet<>();
        for (String id : entityIdsInRelations()) {
            if (!inText.contains(id)) missing.add(id);
        }
        return missing;
    }
}
