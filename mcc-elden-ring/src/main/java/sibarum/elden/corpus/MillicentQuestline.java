package sibarum.elden.corpus;

import java.util.List;

public final class MillicentQuestline {

    private MillicentQuestline() {}

    public static List<Item> items() {
        return List.of(
            Item.of("unalloyed_gold_needle_broken", "Unalloyed Gold Needle (Broken)", ItemCategory.KEY_ITEM,
                "Unalloyed Gold Needle (Broken): A needle of unalloyed gold, the one metal said to repel "
                + "the influence of the outer gods. Crafted by Miquella in his long labor against "
                + "the Rot God that had taken hold of his twin sister Malenia, the needle was "
                + "intended as the instrument by which the rot might be drawn from her flesh "
                + "entirely. It lies broken when first recovered, snapped in the moment Malenia "
                + "bloomed at Aeonia and the scarlet rot overwhelmed every effort to contain it.",
                "Swamp of Aeonia"
            ),

            Item.of("millicents_prosthesis", "Millicent's Prosthesis", ItemCategory.TALISMAN,
                "Millicent's Prosthesis: A prosthetic arm worn by Millicent, one of the rot-born daughters "
                + "spawned from the bloom of Malenia at Aeonia. Like her sisters, Millicent was born "
                + "of the scarlet rot itself and shares in her mother's affliction, but unlike them "
                + "she resists the rot's pull toward becoming a goddess in Malenia's image. The "
                + "prosthesis allows her to wield a blade in defiance of the wasting that consumes "
                + "her remaining limb.",
                "Caelid"
            ),

            Item.of("gowrys_bell_bearing", "Gowry's Bell Bearing", ItemCategory.KEY_ITEM,
                "Gowry's Bell Bearing: The bell bearing of Gowry, a sage of Sellia who has devoted himself "
                + "to the cultivation of the scarlet rot. Beneath his appearance as a kindly healer "
                + "tending to the ailing Millicent, Gowry conspires to see her ripen into a true "
                + "successor to Malenia — a second Rot Goddess to bloom where the first was thwarted. "
                + "The bell bearing, recovered upon his death, releases his stock of rot-tinged "
                + "wares to the merchants of the Roundtable.",
                "Sellia, Town of Sorcery"
            ),

            Item.of("valkyries_prosthesis", "Valkyrie's Prosthesis", ItemCategory.TALISMAN,
                "Valkyrie's Prosthesis: A prosthetic arm modeled on those worn by Malenia herself, fitted "
                + "to Millicent after Gowry repairs the needle and she takes up her blade in earnest. "
                + "The prosthesis grants its bearer the bearing of a valkyrie of the Haligtree — the "
                + "elite warrior-women who served Malenia in her campaign against Radahn — and marks "
                + "the moment Millicent ceases to be a wandering invalid and becomes, briefly, a "
                + "warrior in her own right."
            ),

            Item.of("millicents_set", "Millicent's Set", ItemCategory.ARMOR,
                "Millicent's Set: The traveling garb of Millicent, light cloth wrapped against the warmth "
                + "of the rot that smolders perpetually beneath her skin. The set bears the simple "
                + "elegance of a noblewoman fallen on humbler circumstances, fitting for one who "
                + "walks the long road from Caelid to the Altus Plateau in search of an identity "
                + "not yet decided between her mother's path and her own."
            ),

            Item.of("rotten_winged_sword_insignia", "Rotten Winged Sword Insignia", ItemCategory.TALISMAN,
                "Rotten Winged Sword Insignia: A talisman bearing the winged sword crest of the valkyries "
                + "of the Haligtree, corroded by the rot that infused those who served Malenia "
                + "longest. The talisman is recovered after the trial at the Altus Plateau, where "
                + "Millicent faces her sisters — the other rot-born daughters of Aeonia — and "
                + "chooses, in defiance of Gowry's design, to stand against her own apotheosis as "
                + "the next Rot Goddess.",
                "Shaded Castle"
            ),

            Item.of("scarlet_aeonia", "Scarlet Aeonia", ItemCategory.INCANTATION,
                "Scarlet Aeonia: An incantation reproducing the bloom by which Malenia first unleashed the "
                + "scarlet rot upon Caelid in her duel against Radahn. The flower that grows from "
                + "the wound serves as both weapon and symbol: every petal of Aeonia is a step "
                + "closer to Malenia's full ascension as the Rot Goddess her inner god demands she "
                + "become, and each bloom further withers the land that receives it.",
                "Sister Haligtree"
            ),

            Item.of("remembrance_rot_goddess", "Remembrance of the Rot Goddess", ItemCategory.REMEMBRANCE,
                "Remembrance of the Rot Goddess: The remembrance of Malenia, Blade of Miquella and severed "
                + "twin of the Empyrean himself. Born with the scarlet rot already nested in her "
                + "soul as her inner god, Malenia spent her life in resistance against the bloom "
                + "she carried, serving as her brother's sword in the hope that he might one day "
                + "cure her. Her defeat in the rotted depths of the Haligtree marks the final "
                + "blooming she had so long postponed.",
                "Sister Haligtree"
            ),

            Item.of("hand_of_malenia", "Hand of Malenia", ItemCategory.WEAPON,
                "Hand of Malenia: The katana wielded by Malenia in her prosthetic right hand, forged of a "
                + "metal that does not rust even in the rotted swamps where she made her final "
                + "stand. The blade's signature technique, Waterfowl Dance, was developed by Malenia "
                + "in her training with the swordmasters of the Land of Reeds — a discipline "
                + "preserved within her even as the rot claimed more of her body, and the last "
                + "expression of the warrior she had been before her affliction defined her wholly.",
                "Sister Haligtree"
            )
        );
    }

}
