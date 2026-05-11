package sibarum.elden.annotation;

import java.util.List;

public final class VolcanoManorAnnotations {

    private VolcanoManorAnnotations() {}

    public static List<AnnotatedItem> items() {
        return List.of(
                AnnotationParser.parse(
                        "remembrance_blasphemous",
                        "[Remembrance of the Blasphemous](artifact:remembrance_blasphemous): The "
                                + "[remembrance](concept:remembrance) of [Rykard, Lord of Blasphemy](character:rykard), "
                                + "second son of [Queen Rennala](character:rennala) and [Radagon](character:radagon) "
                                + "and once a [demigod](concept:demigod) of the [Golden Order](concept:golden_order). "
                                + "Rykard renounced the order entirely, allied himself with the "
                                + "[Great Serpent](character:great_serpent) that had long dwelt beneath "
                                + "[Mt. Gelmir](place:mt_gelmir), and sought to become a god through the "
                                + "[endless consumption of heroes](event:rykards_apotheosis) — a path of apotheosis "
                                + "through devouring rather than through grace, in deliberate inversion of every "
                                + "tenet the [Erdtree](place:erdtree) had imposed.",
                        List.of(
                                new Relation("remembrance_blasphemous", RelationType.IS_A,         "remembrance"),
                                new Relation("remembrance_blasphemous", RelationType.EMBODIED,     "rykard"),
                                new Relation("rykard",                  RelationType.IS_A,         "demigod"),
                                new Relation("rykard",                  RelationType.KIN_TO,       "rennala"),
                                new Relation("rykard",                  RelationType.KIN_TO,       "radagon"),
                                new Relation("rykard",                  RelationType.PART_OF,      "golden_order"),
                                new Relation("rykard",                  RelationType.REVERSED,     "golden_order"),
                                new Relation("rykard",                  RelationType.PART_OF,      "great_serpent"),
                                new Relation("great_serpent",           RelationType.AT_LOCATION,  "mt_gelmir"),
                                new Relation("rykard",                  RelationType.AGENT_OF,     "rykards_apotheosis"),
                                new Relation("rykards_apotheosis",      RelationType.REVERSED,     "erdtree"),
                                new Relation("rykards_apotheosis",      RelationType.FAILED,       "rykard"),
                                new Relation("rykard",                  RelationType.CAUSED,       "rykards_apotheosis"),
                                new Relation("elden_ring_shattering",   RelationType.CAUSED,       "rykards_apotheosis"),
                                new Relation("rykards_apotheosis",      RelationType.CAUSED,       "rykards_death"),
                                new Relation("rykards_death",           RelationType.CAUSED,       "remembrance_blasphemous")
                        )
                ),

                AnnotationParser.parse(
                        "tonic_of_forgetfulness",
                        "[Tonic of Forgetfulness](artifact:tonic_of_forgetfulness): A draught provided by "
                                + "[Tanith](character:tanith) that, when administered, erases a specific memory "
                                + "from its drinker. Tanith offers the tonic to be given to [Rya](character:rya), "
                                + "her attendant and emissary, so that Rya might remain unaware of her own true "
                                + "nature as a [serpent-woman](concept:serpent_woman) of [Rykard](character:rykard)'s "
                                + "[brood](concept:rykards_brood). The gesture is presented as a mercy — that Rya "
                                + "may continue to live as the noblewoman she believes herself to be — though it "
                                + "is equally a means of keeping a useful servant bound to the [manor](place:volcano_manor) "
                                + "in ignorance of what her lord has made of his household.",
                        List.of(
                                new Relation("tanith",                  RelationType.CREATED,      "tonic_of_forgetfulness"),
                                new Relation("tonic_of_forgetfulness",  RelationType.USED_IN,      "ryas_deception"),
                                new Relation("tanith",                  RelationType.AGENT_OF,     "ryas_deception"),
                                new Relation("ryas_deception",          RelationType.AT_LOCATION,  "volcano_manor"),
                                new Relation("rya",                     RelationType.IS_A,         "serpent_woman"),
                                new Relation("rya",                     RelationType.PART_OF,      "rykards_brood"),
                                new Relation("rykard",                  RelationType.CREATED,      "rykards_brood"),
                                new Relation("tanith",                  RelationType.KIN_TO,       "rykard"),
                                new Relation("tanith",                  RelationType.CAUSED,       "ryas_deception"),
                                new Relation("tonic_of_forgetfulness",  RelationType.CAUSED,       "ryas_memory_loss")
                        )
                )
        );
    }
}
