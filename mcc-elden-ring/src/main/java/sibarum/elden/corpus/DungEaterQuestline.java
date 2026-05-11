package sibarum.elden.corpus;

import java.util.List;

public final class DungEaterQuestline {



    public static List<Item> items() {
        return List.of(
                Item.of("seedbed_curse", "Seedbed Curse", ItemCategory.KEY_ITEM,
                        "Seedbed Curse: A curse left in the corpse of one who has been violated by the Dung Eater. "
                                + "The curse takes root in the soul of its host and propagates through the cycle of "
                                + "death and return, ensuring that those marked by it can never know peace in the "
                                + "Erdtree's embrace — their souls forever soiled, forever cast out from any "
                                + "afterlife the Golden Order would offer. The Dung Eater pursues his work with "
                                + "methodical patience, sowing curses across the Lands Between as one might plant "
                                + "a long-growing crop."
                ),

                Item.of("sewer_gaol_key", "Sewer-Gaol Key", ItemCategory.KEY_ITEM,
                        "Sewer-Gaol Key: A key to the cell in the Subterranean Shunning-Grounds beneath Leyndell "
                                + "where the Dung Eater's true body is held. The shackled prisoner is the original "
                                + "from which the armored figure roaming the capital is merely a projection — a "
                                + "soiled body kept locked away by the Erdtree's keepers, who could neither destroy "
                                + "him nor permit his release, and who understood that his curse would propagate "
                                + "regardless of whether his flesh walked free.",
                        "Subterranean Shunning-Grounds"
                ),

                Item.of("fingerprint_stone_shield", "Fingerprint Stone Shield", ItemCategory.SHIELD,
                        "Fingerprint Stone Shield: An enormous greatshield of unworked stone bearing the "
                                + "impressions of countless fingers, wielded by the Dung Eater in his armored "
                                + "incarnation. The marks upon it are said to be those of the dead whose corpses "
                                + "he defiled, each finger a record of a soul he set upon the path of the seedbed "
                                + "curse — the shield itself thus a ledger of his work, carried into battle as "
                                + "both weapon and accounting."
                ),

                Item.of("dung_eaters_set", "Dung Eater's Set", ItemCategory.ARMOR,
                        "Dung Eater's Set: Heavy plate worn by the Dung Eater as he stalks the streets of Leyndell "
                                + "in search of fresh corpses. The armor is caked in filth that no cleaning would "
                                + "remove, and bears the iconography of the Erdtree turned to mockery — for the "
                                + "Dung Eater understands himself as the Golden Order's necessary inverse, the "
                                + "proof by negation that the order's promise of clean death and rebirth is a lie "
                                + "that requires souls like his to expose.",
                        "Leyndell, Royal Capital"
                ),

                Item.of("mending_rune_fell_curse", "Mending Rune of the Fell Curse", ItemCategory.KEY_ITEM,
                        "Mending Rune of the Fell Curse: A mending rune produced by the consumption of seedbed "
                                + "curses gathered from across the Lands Between. When woven into the Elden Ring, "
                                + "this rune institutes an age in which every soul of the Lands Between is cursed "
                                + "as the Dung Eater himself is — a final equality achieved not through grace but "
                                + "through universal defilement, such that the distinction between the blessed and "
                                + "the soiled is dissolved in favor of the latter."
                )
        );
    }
}
