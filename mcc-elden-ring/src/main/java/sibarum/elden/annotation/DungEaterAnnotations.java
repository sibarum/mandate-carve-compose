package sibarum.elden.annotation;

import java.util.List;

public final class DungEaterAnnotations {

    private DungEaterAnnotations() {}

    public static List<AnnotatedItem> items() {
        return List.of(
                AnnotationParser.parse(
                        "mending_rune_fell_curse",
                        "[Mending Rune of the Fell Curse](artifact:mending_rune_fell_curse): A "
                                + "[mending rune](concept:mending_rune) produced by the consumption of "
                                + "[seedbed curses](concept:seedbed_curse) gathered from across the "
                                + "[Lands Between](place:lands_between). When woven into the "
                                + "[Elden Ring](artifact:elden_ring), this rune institutes an "
                                + "[age of the fell curse](event:age_of_the_fell_curse) in which every soul of "
                                + "the Lands Between is cursed as the [Dung Eater](character:dung_eater) himself "
                                + "is — a final equality achieved not through grace but through universal "
                                + "defilement, such that the distinction between the blessed and the soiled is "
                                + "dissolved in favor of the latter.",
                        List.of(
                                new Relation("mending_rune_fell_curse",  RelationType.IS_A,        "mending_rune"),
                                new Relation("seedbed_curse",                   RelationType.AGENT_OF,    "soul_defilement"),
                                new Relation("dung_eater",                      RelationType.CREATED,     "seedbed_curse"),
                                new Relation("seedbed_curse",                   RelationType.USED_IN,     "mending_rune_fell_curse"),
                                new Relation("mending_rune_fell_curse",  RelationType.USED_IN,     "elden_ring"),
                                new Relation("mending_rune_fell_curse",  RelationType.ENABLED,     "age_of_the_fell_curse"),
                                new Relation("age_of_the_fell_curse",           RelationType.AT_LOCATION, "lands_between"),
                                new Relation("age_of_the_fell_curse",           RelationType.REVERSED,    "golden_order")
                        )
                )
        );
    }
}
