package sibarum.elden.corpus;

import java.util.List;

public final class RannisQuestline {

    private RannisQuestline() {}

    public static List<Item> items() {
        return List.of(
                Item.of("carian_inverted_statue", "Carian Inverted Statue", ItemCategory.KEY_ITEM,
                        "Carian Inverted Statue: A relic of House Caria depicting a figure in inverted posture, used "
                                + "to flip the orientation of the Divine Tower of Liurnia. The mechanism was devised "
                                + "so that one might traverse the tower in defiance of its original purpose, granting "
                                + "access to chambers sealed away during the schism between the Carian royal family "
                                + "and the Golden Order. Ranni, exiled daughter of Rennala and Radagon, made use of "
                                + "this artifact during her pursuit of the path that would lead her beyond the "
                                + "Greater Will's influence."
                ),
                Item.of("fingerslayer_blade", "Fingerslayer Blade", ItemCategory.WEAPON,
                        "Fingerslayer Blade: A legendary weapon hidden deep beneath Nokron, the Eternal City, capable "
                                + "of slaying the Two Fingers themselves — the messengers through which the Greater "
                                + "Will communicates with the Lands Between. Forged in an age before the Erdtree, "
                                + "when the Eternal Cities still flourished under the stars, the blade was sought by "
                                + "Ranni as the instrument required to sever the tether between herself and her "
                                + "appointed Two Fingers, an act foundational to her plan of authoring a new age free "
                                + "of outer god intervention.",
                        "Nokron, the Eternal City"
                ),
                Item.of("dark_moon_ring", "Dark Moon Ring", ItemCategory.TALISMAN,
                        "Dark Moon Ring: A ring bearing the cold light of the Dark Moon, exchanged in a ceremony of "
                                + "betrothal at the summit of the Cathedral of Manus Celes. To place this ring upon "
                                + "the finger of Ranni is to consent to becoming her consort and lord, accompanying "
                                + "her into the far-flung age she intends to bring about — an age of stars and "
                                + "certainty kept at a remove from the affairs of life.",
                        "Cathedral of Manus Celes"
                ),
                Item.of("rannis_dark_moon", "Ranni's Dark Moon", ItemCategory.SORCERY,
                        "Ranni's Dark Moon: A sorcery embodying the full power of the Dark Moon, granted by Ranni "
                                + "upon the completion of her great working. The spell summons a frigid moon that "
                                + "descends upon foes and strips them of their magical defenses, reflecting the "
                                + "nature of Ranni's chosen age: one in which the warmth and certainty of the Golden "
                                + "Order are replaced by the cold, distant clarity of the moon and stars."
                ),
                Item.of("cursemark_of_death", "Cursemark of Death", ItemCategory.KEY_ITEM,
                        "Cursemark of Death: A mark borne by those of the Empyrean lineage who carry within them the "
                                + "stolen Rune of Death. Ranni inherited this cursemark from her mother's bloodline "
                                + "and, on the Night of the Black Knives, used it as the basis for the ritual that "
                                + "produced the assassins' daggers. The mark allowed her to slay her own demigod "
                                + "flesh and abandon her body — a necessary step in escaping the destiny imposed "
                                + "upon Empyreans by the Two Fingers, though her soul has wandered ever since in "
                                + "search of a vessel and a path forward."
                ),
                Item.of("discarded_palace_key", "Discarded Palace Key", ItemCategory.KEY_ITEM,
                        "Discarded Palace Key: A key once belonging to Ranni, cast away in the snowfields beyond "
                                + "the Consecrated Snowfield. The key unlocks her chambers within Renna's Rise, where "
                                + "her doll-vessel awaits the Tarnished who has completed her labors. That she left "
                                + "the key to be found, rather than concealed, reflects her trust in the chosen "
                                + "Tarnished who would carry out the works that her own diminished form could not.",
                        "Consecrated Snowfield"
                ),
                Item.of("black_knifeprint", "Black Knifeprint", ItemCategory.KEY_ITEM,
                        "Black Knifeprint: An impression of one of the daggers used on the Night of the Black "
                                + "Knives, recovered from the corpse of an assassin. The print serves as evidence "
                                + "of the conspiracy and points toward its true architect; presented to certain "
                                + "figures still loyal to the memory of Godwyn, it confirms what had long been "
                                + "suspected — that the orchestrator of that night was not an outside enemy but a "
                                + "daughter of the royal house itself, who chose the murder of her half-brother as "
                                + "the price of her own liberation from divine appointment."
                ),
                Item.of("remembrance_baleful_shadow", "Remembrance of the Baleful Shadow", ItemCategory.REMEMBRANCE,
                        "Remembrance of the Baleful Shadow: The shade dispatched by the Two Fingers to assassinate "
                                + "Ranni in retribution for the Fingerslayer's use. Defeating this shadow, which "
                                + "wears the form of Blaidd, Ranni's loyal wolf-shadow and half-brother, severs the "
                                + "final means by which the Greater Will might reach her — and seals Blaidd's tragic "
                                + "fate as one bound by oath to a mistress whose chosen path leaves no place for him.",
                        "Defeating the Baleful Shadow"
                )
        );
    }
}
