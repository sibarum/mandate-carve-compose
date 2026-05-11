package sibarum.elden.annotation;

import java.util.List;

public final class MillicentAnnotations {

    public static List<AnnotatedItem> items() {
        return List.of(
                AnnotationParser.parse(
                        "scarlet_aeonia",
                        "[Scarlet Aeonia](artifact:scarlet_aeonia): An [incantation](concept:incantation) reproducing "
                                + "the bloom by which [Malenia](character:malenia) first unleashed the "
                                + "[scarlet rot](concept:scarlet_rot) upon [Caelid](place:caelid) in her "
                                + "[duel against Radahn](event:battle_of_aeonia). The flower that grows from the wound "
                                + "serves as both weapon and symbol: every petal of Aeonia is a step closer to "
                                + "Malenia's full ascension as the [Rot Goddess](concept:rot_god) her "
                                + "[inner god](concept:inner_god) demands she become, and each bloom further withers "
                                + "the land that receives it.",
                        List.of(
                                new Relation("scarlet_aeonia",     RelationType.IS_A,          "incantation"),
                                new Relation("malenia",            RelationType.CREATED,       "scarlet_aeonia"  ),
                                new Relation("scarlet_aeonia",     RelationType.EMBODIED,      "scarlet_rot"),
                                new Relation("rot_god",            RelationType.IS_A,          "inner_god"),
                                new Relation("malenia",            RelationType.EMBODIED,      "rot_god"),
                                new Relation("battle_of_aeonia",   RelationType.AT_LOCATION,   "caelid"),
                                new Relation("malenia",            RelationType.AGENT_OF,      "battle_of_aeonia"),
                                new Relation("battle_of_aeonia",   RelationType.CAUSED,        "scarlet_rot"),
                                new Relation("scarlet_rot",        RelationType.CAUSED,        "caelid_corruption")
                        )
                )
        );
    }



}
