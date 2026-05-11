package sibarum.elden.annotation;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Type declarations for entities that exist in the causal graph (referenced in
 * relations) but are never marked as spans in any prose. These are "inferred"
 * entities — verbal nouns like "Marika's grief" or "Godfrey's exile" that the
 * descriptions express implicitly rather than as named tokens.
 *
 * Spans (in {@link AnnotatedItem}) are the training signal for Part 1, which
 * learns to extract typed entity mentions from raw text. Implicit declarations
 * are NOT training signal — they're metadata about the knowledge graph used
 * for inspection and as type information for Part 2's reasoning nodes.
 *
 * If a relation references an entity id that appears nowhere — not in a span
 * and not in this map — the graph builder will still accept it but report it
 * as untyped.
 */
public final class ImplicitEntities {

    private ImplicitEntities() {}

    public static Map<String, EntityType> shatteringEra() {
        Map<String, EntityType> m = new LinkedHashMap<>();

        // Events — discrete things that happened.
        m.put("elden_ring_shattering",          EntityType.EVENT);
        m.put("godfrey_exile",                  EntityType.EVENT);
        m.put("marikas_grief",                  EntityType.EVENT);
        m.put("godwyn_death_of_soul",           EntityType.EVENT);
        m.put("erdtree_corruption",             EntityType.EVENT);
        m.put("fortissax_corruption",           EntityType.EVENT);
        m.put("golden_order_destabilization",   EntityType.EVENT);
        m.put("golden_order_persecution",       EntityType.EVENT);
        m.put("destined_death_removal",         EntityType.EVENT);
        m.put("radagon_repair_attempt",         EntityType.EVENT);
        m.put("miquella_disillusionment",       EntityType.EVENT);
        m.put("miquella_abandons_fundamentalism", EntityType.EVENT);
        m.put("miquella_restoration_attempt",   EntityType.EVENT);
        m.put("malenia_scarlet_rot",            EntityType.EVENT);

        // Concepts / states — ongoing or abstract.
        m.put("marikas_burden",                 EntityType.CONCEPT);
        m.put("deathroot",                      EntityType.CONCEPT);

        // Eras.
        m.put("age_of_duskborn",                EntityType.ERA);

        // Artifacts referenced but never marked up.
        m.put("cursemark_of_death",             EntityType.ARTIFACT);

        return Map.copyOf(m);
    }

    public static Map<String, EntityType> millicentQuestline() {
        return Map.of(
                "caelid_corruption",            EntityType.EVENT
        );
    }

    public static Map<String, EntityType> dungEaterQuestline() {
        return Map.of(
                "soul_defilement",              EntityType.EVENT
        );
    }

    public static Map<String, EntityType> volcanoManor() {
        return Map.of(
                "rykards_death",                EntityType.EVENT,
                "ryas_deception",               EntityType.EVENT,
                "ryas_memory_loss",             EntityType.EVENT
        );
    }

    public static Map<String, EntityType> all() {
        LinkedHashMap<String, EntityType> m = new LinkedHashMap<>();
        m.putAll(shatteringEra());
        m.putAll(millicentQuestline());
        m.putAll(dungEaterQuestline());
        m.putAll(volcanoManor());
        return Map.copyOf(m);
    }
}
