package sibarum.elden.annotation;

import java.util.List;

/**
 * Hand-authored annotations for the 10 Shattering-era items in
 * {@link sibarum.elden.corpus.ShatteringEra}. Each annotated item is the
 * supervision signal Part 1 (the entity-and-relation extractor) trains on.
 *
 * Authoring guide:
 *
 *   1st arg: item id (must match a corpus item).
 *   2nd arg: the description text WITH inline entity markup:
 *               [surface form](type:entity_id)
 *            where `type` is an EntityType (character, artifact, event, place,
 *            faction, era, concept) and `entity_id` is a free-form snake_case
 *            canonical name. Reuse the same id across items so the cross-item
 *            graph builds itself.
 *   3rd arg: list of typed relations between entity ids. Entity ids in
 *            relations don't have to appear in the text — that's how you declare
 *            structural truths the prose doesn't surface.
 *
 * The first entry is fully annotated as a worked example. Annotate the other
 * nine in the same style.
 */
public final class ShatteringEraAnnotations {

    private ShatteringEraAnnotations() {}

    public static List<AnnotatedItem> items() {
        return List.of(

                // ============================================================
                // Worked example.
                // ============================================================
                AnnotationParser.parse(
                        "black_knife",
                        "[Black Knife](artifact:black_knife): A ritual dagger imbued with a stolen fragment of the [Rune of Death](artifact:rune_of_death) — "
                                + "specifically only the soul-killing half — by [Ranni the Witch](character:ranni), who "
                                + "commissioned the [Night of the Black Knives](event:night_of_black_knives). The power "
                                + "sealed within these blades had long been locked away by [Queen Marika](character:marika), "
                                + "and their use on [Godwyn the Golden](character:godwyn) resulted in an incomplete death: "
                                + "his soul perished while his body lived on, setting in motion a chain of corruption that "
                                + "would spread through the [Erdtree](place:erdtree)'s roots and eventually destabilize the "
                                + "entire [Golden Order](concept:golden_order).",
                        List.of(
                                new Relation("ranni",                       RelationType.CAUSED,   "night_of_black_knives"),
                                new Relation("rune_of_death",               RelationType.PART_OF, "black_knife"),
                                new Relation("black_knife",                 RelationType.PART_OF, "night_of_black_knives"),
                                new Relation("night_of_black_knives",       RelationType.CAUSED,   "godwyn_death_of_soul"),
                                new Relation("godwyn_death_of_soul",        RelationType.CAUSED,   "erdtree_corruption"),
                                new Relation("erdtree_corruption",          RelationType.CAUSED,   "golden_order_destabilization")
                        )
                ),

        AnnotationParser.parse(
                "remembrance_lichdragon",
                "[Remembrance of the Lichdragon](artifact:remembrance_lichdragon): After [Godwyn the Golden](character:godwyn) became the "
                        + "[Prince of Death](concept:prince_of_death), the ancient dragon "
                        + "[Fortissax](character:fortissax) fought long and hard against the "
                        + "[Death](concept:destined_death) spreading within his companion. Alas, victory "
                        + "was never achieved, and its only reward was corruption. This remembrance encodes "
                        + "Fortissax's vigil — an intimate record of how [Godwyn's half-death](event:godwyn_death_of_soul) "
                        + "was not simply a wound but an active, spreading force that consumed even those "
                        + "who tried to resist it from within.",
                List.of(
                        new Relation("godwyn_death_of_soul",  RelationType.CAUSED,    "prince_of_death"),
                        new Relation("fortissax",             RelationType.RESISTED,  "prince_of_death"),
                        new Relation("prince_of_death",       RelationType.CAUSED,    "fortissax_corruption"),
                        new Relation("fortissax_corruption",  RelationType.PART_OF,  "erdtree_corruption")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "eclipse_shotel",
                "[Eclipse Shotel](artifact:eclipse_shotel): A storied blade and treasure of [Castle Sol](place:castle_sol), shaped in the image of "
                        + "an eclipsed sun drained of color. The eclipse it depicts is not a solar phenomenon "
                        + "but a theological one — the dimming of life by the "
                        + "[Prince of Death](concept:prince_of_death)'s power. Its weapon art, Death Flare, "
                        + "sets the lusterless sun ablaze with the [Prince of Death](concept:prince_of_death)'s "
                        + "flames, inflicting the death ailment upon foes. The blade stands as a relic of the "
                        + "cult that formed around [Godwyn](character:godwyn)'s half-dead state, worshipping "
                        + "the eclipse as an emblem of death's new foothold in the world following the "
                        + "[Night of the Black Knives](event:night_of_black_knives).",
                List.of(
                        new Relation("godwyn_death_of_soul",  RelationType.CAUSED,    "prince_of_death"),
                        new Relation("night_of_black_knives", RelationType.CAUSED,    "prince_of_death"),
                        new Relation("prince_of_death",       RelationType.CAUSED,    "those_who_live_in_death"),
                        new Relation("eclipse_shotel",        RelationType.EMBODIED, "prince_of_death")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "marikas_hammer",
                "[Marika's Hammer](artifact:marikas_hammer): A stone hammer crafted in the Numen lands outside the Lands Between — the very tool with "
                        + "which [Queen Marika](character:marika) shattered the [Elden Ring](artifact:elden_ring) "
                        + "and with which [Radagon](character:radagon) later attempted to repair it. The hammer "
                        + "partially broke upon the shattering, becoming splintered with rune fragments. That "
                        + "both acts — destruction and desperate repair — were performed with the same instrument "
                        + "speaks to the contradictory nature of [Marika](character:marika) and "
                        + "[Radagon](character:radagon) as two halves of one being, one yielding to grief at "
                        + "[Godwyn](character:godwyn)'s death, the other straining against it.",
                List.of(
                        new Relation("godwyn_death_of_soul",  RelationType.CAUSED,   "marikas_grief"),
                        new Relation("marikas_grief",         RelationType.CAUSED,   "elden_ring_shattering"),
                        new Relation("marika",                RelationType.CAUSED,   "elden_ring_shattering"),
                        new Relation("marikas_hammer",        RelationType.USED_IN, "elden_ring_shattering"),
                        new Relation("radagon",               RelationType.RESISTED, "elden_ring_shattering"),
                        new Relation("marikas_hammer",        RelationType.USED_IN, "radagon_repair_attempt"),
                        new Relation("radagon_repair_attempt",RelationType.FAILED,  "elden_ring_shattering")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "godfrey_icon",
                "[Godfrey Icon](artifact:godfrey_icon): A legendary talisman depicting [Elden Lord Godfrey](character:godfrey), the first "
                        + "[Elden Lord](concept:elden_lord) and [Queen Marika](character:marika)'s original "
                        + "consort. [Godfrey](character:godfrey) was a ferocious warrior who, when he vowed "
                        + "to become a lord, took the Beast Regent [Serosh](character:serosh) upon his back "
                        + "to suppress the ceaseless lust for battle that raged within. The talisman "
                        + "memorializes the golden lineage — [Godfrey](character:godfrey) and "
                        + "[Marika](character:marika)'s bloodline, which produced "
                        + "[Godwyn the Golden](character:godwyn) among others — whose eventual exile from "
                        + "the [Lands Between](place:lands_between) left the demigod hierarchy without its "
                        + "stabilizing patriarch, a vacuum that would deepen the chaos of "
                        + "the [Shattering](event:elden_ring_shattering).",
                List.of(
                        new Relation("godfrey",              RelationType.KIN_TO, "godwyn"),
                        new Relation("godfrey",              RelationType.KIN_TO,"marika"),
                        new Relation("marika",               RelationType.CAUSED,     "godfrey_exile"),
                        new Relation("godfrey_exile",        RelationType.CAUSED,     "golden_order_destabilization"),
                        new Relation("golden_order_destabilization", RelationType.CAUSED, "elden_ring_shattering")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "marikas_soreseal",
                "[Marika's Soreseal](artifact:marikas_soreseal): A legendary talisman in the form of an eye engraved with an Elden Rune, said to be the "
                        + "seal of [Queen Marika](character:marika). It greatly raises the wearer's arcane "
                        + "faculties but increases damage taken by a comparable measure — solemn duty weighs "
                        + "upon the one beholden, not unlike a gnawing curse from which there is no "
                        + "deliverance. Found deep within [Miquella](character:miquella)'s "
                        + "[Haligtree](place:haligtree), the seal suggests that the divine burden "
                        + "[Marika](character:marika) carried — sustaining the [Golden Order](concept:golden_order), "
                        + "containing [Destined Death](concept:destined_death), holding the "
                        + "[Elden Ring](artifact:elden_ring) — was itself a kind of affliction, and that "
                        + "the cost of that burden ultimately contributed to her breaking under the weight "
                        + "of [Godwyn](character:godwyn)'s death.",
                List.of(
                        new Relation("marika",               RelationType.BORE,     "golden_order"),
                        new Relation("marika",               RelationType.BORE,     "elden_ring"),
                        new Relation("marikas_soreseal",     RelationType.EMBODIED, "marikas_burden"),
                        new Relation("marikas_burden",       RelationType.CAUSED,    "marikas_grief"),
                        new Relation("godwyn_death_of_soul", RelationType.CAUSED,    "marikas_grief"),
                        new Relation("marikas_grief",        RelationType.CAUSED,    "elden_ring_shattering")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "triple_rings_of_light",
                "[Triple Rings of Light](artifact:triple_rings_of_light): One of the incantations of the [Golden Order](concept:golden_order) fundamentalists, "
                        + "this spell produces three rings of holy light and fires them outward before they "
                        + "arc back toward the caster. It was a gift from the young "
                        + "[Miquella](character:miquella) to his father [Radagon](character:radagon), "
                        + "placing [Miquella](character:miquella) squarely within the fundamentalist tradition "
                        + "before his disillusionment. His later abandonment of "
                        + "[Golden Order Fundamentalism](concept:golden_order_fundamentalism) — when it failed "
                        + "to treat [Malenia](character:malenia)'s rot or reverse "
                        + "[Godwyn](character:godwyn)'s fate — stands in direct contrast to this early act "
                        + "of filial devotion to [Radagon](character:radagon) and the "
                        + "[Golden Order](concept:golden_order).",
                List.of(
                        new Relation("miquella",                        RelationType.AGENT_OF,      "golden_order_fundamentalism"),
                        new Relation("miquella",                        RelationType.CREATED,       "triple_rings_of_light"),
                        new Relation("triple_rings_of_light",           RelationType.PART_OF,       "golden_order_fundamentalism"),
                        new Relation("godwyn_death_of_soul",            RelationType.CAUSED,        "miquella_disillusionment"),
                        new Relation("malenia_scarlet_rot",             RelationType.CAUSED,        "miquella_disillusionment"),
                        new Relation("miquella_disillusionment",        RelationType.CAUSED,        "miquella_abandons_fundamentalism"),
                        new Relation("miquella_abandons_fundamentalism",RelationType.CAUSED,        "miquella_restoration_attempt")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "orders_blade",
                "[Order's Blade](artifact:orders_blade): One of the incantations of the [Golden Order](concept:golden_order) fundamentalists, "
                        + "used by hunters of [Those Who Live in Death](concept:those_who_live_in_death). "
                        + "It enchants the caster's weapon with holy power especially damaging to the "
                        + "undead, and any felled by it cannot be revived. The role of such hunters is "
                        + "to stamp out defiled reason — all for the perfection of the "
                        + "[Golden Order](concept:golden_order). The existence of this incantation is itself "
                        + "a consequence of [Godwyn](character:godwyn)'s half-death: once his corrupted body "
                        + "began seeding undeath through the [Erdtree](place:erdtree)'s roots in the form of "
                        + "[Deathroot](concept:deathroot), the [Golden Order](concept:golden_order) was forced "
                        + "to institutionalize the persecution of "
                        + "[Those Who Live in Death](concept:those_who_live_in_death), treating the symptom "
                        + "while refusing to address the cause.",
                List.of(
                        new Relation("godwyn_death_of_soul",   RelationType.CAUSED,    "erdtree_corruption"),
                        new Relation("erdtree_corruption",     RelationType.CAUSED,    "deathroot"),
                        new Relation("deathroot",              RelationType.CAUSED,    "those_who_live_in_death"),
                        new Relation("those_who_live_in_death",RelationType.CAUSED,    "golden_order_persecution"),
                        new Relation("orders_blade",           RelationType.USED_IN,  "golden_order_persecution"),
                        new Relation("golden_order_persecution",RelationType.PART_OF, "golden_order_destabilization")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "unalloyed_gold_needle",
                "[Unalloyed Gold Needle](artifact:unalloyed_gold_needle): [Miquella](character:miquella) crafted needles of unalloyed gold capable of warding away "
                        + "the meddling of outer gods and forestalling "
                        + "[Malenia](character:malenia)'s incurable [Scarlet Rot](concept:scarlet_rot). "
                        + "[Unalloyed gold](concept:unalloyed_gold) — gold untouched by the influence of any "
                        + "outer god, including the [Greater Will](concept:greater_will) — was "
                        + "[Miquella](character:miquella)'s answer to a theological problem: the "
                        + "[Golden Order](concept:golden_order)'s god-endorsed framework could not cure "
                        + "ailments caused by other gods. These needles represent "
                        + "[Miquella](character:miquella)'s pivot away from "
                        + "[Golden Order Fundamentalism](concept:golden_order_fundamentalism) toward an "
                        + "independent project of healing, and hint at his broader ambition to build an "
                        + "order free from divine coercion — the seed of the "
                        + "[Haligtree](place:haligtree).",
                List.of(
                        new Relation("malenia_scarlet_rot",             RelationType.CAUSED,        "miquella_disillusionment"),
                        new Relation("miquella_disillusionment",        RelationType.CAUSED,        "miquella_abandons_fundamentalism"),
                        new Relation("miquella_abandons_fundamentalism",RelationType.CAUSED,        "unalloyed_gold_needle"),
                        new Relation("miquella",                        RelationType.CREATED,       "unalloyed_gold_needle"),
                        new Relation("unalloyed_gold_needle",           RelationType.RESISTED,      "greater_will"),
                        new Relation("unalloyed_gold_needle",           RelationType.CAUSED,        "haligtree"),
                        new Relation("haligtree",                       RelationType.PART_OF,       "miquella_restoration_attempt")
                )
        ),

// ============================================================

        AnnotationParser.parse(
                "mending_rune_death_prince",
                "[Mending Rune of the Death-Prince](artifact:mending_rune_death_prince): A rune gestated by [Fia](character:fia), the Deathbed Companion, formed from the two "
                        + "hallowbrand half-wheels combined — the [Cursemark of Death](artifact:cursemark_of_death) "
                        + "split when both [Godwyn](character:godwyn) and [Ranni](character:ranni) perished "
                        + "simultaneously on the [Night of the Black Knives](event:night_of_black_knives). "
                        + "Used to restore the fractured [Elden Ring](artifact:elden_ring), it would embed "
                        + "the principle of life within death into the new Order. Where "
                        + "[Marika](character:marika) excised [Destined Death](concept:destined_death) from "
                        + "the [Elden Ring](artifact:elden_ring) to create a world of enforced immortality, "
                        + "this rune would reintegrate it — acknowledging "
                        + "[Those Who Live in Death](concept:those_who_live_in_death) not as abominations "
                        + "but as a natural consequence of death's wrongful exile, and offering an answer to "
                        + "the wound first opened by the [Black Knives](artifact:black_knife).",
                List.of(
                        new Relation("night_of_black_knives",       RelationType.CAUSED,    "cursemark_of_death"),
                        new Relation("cursemark_of_death",          RelationType.PART_OF,  "mending_rune_death_prince"),
                        new Relation("fia",                         RelationType.CREATED,  "mending_rune_death_prince"),
                        new Relation("marika",                      RelationType.CAUSED,    "destined_death_removal"),
                        new Relation("destined_death_removal",      RelationType.CAUSED,    "those_who_live_in_death"),
                        new Relation("mending_rune_death_prince", RelationType.REVERSED, "destined_death_removal"),
                        new Relation("mending_rune_death_prince", RelationType.CAUSED,   "age_of_duskborn")
                )
        ));
    };
}
