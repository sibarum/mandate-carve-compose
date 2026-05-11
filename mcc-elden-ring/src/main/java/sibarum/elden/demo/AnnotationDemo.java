package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.EntitySpan;
import sibarum.elden.annotation.Relation;
import sibarum.elden.annotation.ShatteringEraAnnotations;

import java.util.List;

public final class AnnotationDemo {

    public static void main(String[] args) {
        List<AnnotatedItem> items = ShatteringEraAnnotations.items();

        System.out.println("Annotated items: " + items.size());
        System.out.println();

        for (AnnotatedItem item : items) {
            System.out.println("item: " + item.itemId());
            System.out.println("  raw text:");
            System.out.println("    " + item.rawText());
            System.out.println("  spans (" + item.spans().size() + "):");
            for (EntitySpan span : item.spans()) {
                System.out.printf("    [%d..%d] %-9s %-30s -> %s%n",
                        span.start(), span.end(), span.type(), '"' + span.surface() + '"', span.entityId());
            }
            System.out.println("  relations (" + item.relations().size() + "):");
            for (Relation r : item.relations()) {
                System.out.printf("    %-32s --%s--> %s%n", r.subjectId(), r.type(), r.objectId());
            }
            if (!item.entitiesReferencedButNotInText().isEmpty()) {
                System.out.println("  entities in relations but not in text:");
                for (String id : item.entitiesReferencedButNotInText()) {
                    System.out.println("    " + id);
                }
            }
            System.out.println();
        }
    }
}
