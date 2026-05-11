package sibarum.elden.corpus;

import java.util.List;

/**
 * Starter corpus of 10 Shattering-era item slots. Item descriptions are intentionally
 * left as TODO placeholders — the prose from the game is copyrighted, and demo training
 * data should be either (a) your own paraphrased summaries of the canonical text, or
 * (b) clearly-attributed synthetic descriptions written for this project.
 *
 * The 10 slots were chosen to span the key Shattering causal chain:
 *   Night of the Black Knives -> Godwyn's death -> Marika's grief ->
 *   Marika shatters the Elden Ring -> Demigods scatter with Great Runes ->
 *   Shattering War -> Miquella's attempt to restore Godwyn.
 *
 * Replace each TODO with the description text you want the extractor trained on.
 */
public final class ShatteringEra {

    private ShatteringEra() {}

    private static final String TODO = "TODO: fill in description";

    public static List<Item> items() {
        return List.of(                Item.of("black_knife",                  "Black Knife",                       ItemCategory.WEAPON,
                        "A ritual dagger imbued with a stolen fragment of the Rune of Death — specifically only the soul-killing half — by Ranni the Witch, who commissioned the Night of the Black Knives. The power sealed within these blades had long been locked away by Queen Marika, and their use on Godwyn the Golden resulted in an incomplete death: his soul perished while his body lived on, setting in motion a chain of corruption that would spread through the Erdtree's roots and eventually destabilize the entire Golden Order."
                        , "Ringleader's Evergaol; Black Knife Assassin drops"),
                Item.of("mending_rune_death_prince",    "Mending Rune of the Death-Prince",  ItemCategory.KEY_ITEM,
                        "A rune gestated by Fia, the Deathbed Companion, formed from the two hallowbrand half-wheels combined. Used to restore the fractured Elden Ring when brandished by the Elden Lord, it would embed the principle of life within death into the new Order. The Golden Order was created by confining Destined Death — this new Order would be one of Death restored. The rune is Fia's answer to Marika's original act of erasure: where Marika excised death from the Elden Ring to create a world of enforced immortality, the Mending Rune of the Death-Prince would reintegrate it, acknowledging Those Who Live in Death not as abominations but as a natural consequence of death's wrongful exile."
                ),
                Item.of("remembrance_lichdragon",       "Remembrance of the Lichdragon",     ItemCategory.REMEMBRANCE,
                        "After Godwyn the Golden became the Prince of Death, the ancient dragon Fortissax fought long and hard against the Death spreading within his companion. Alas, victory was never achieved, and its only reward was corruption. This remembrance encodes Fortissax's vigil — an intimate record of how Godwyn's half-death was not simply a wound but an active, spreading force that consumed even those who tried to resist it from within."
                        , "Defeating Lichdragon Fortissax"),
                Item.of("eclipse_shotel",               "Eclipse Shotel",                    ItemCategory.WEAPON,
                        "A storied blade and treasure of Castle Sol, shaped in the image of an eclipsed sun drained of color. The eclipse it depicts is not a solar phenomenon but a theological one — the dimming of life by the Prince of Death's power. Its weapon art, Death Flare, sets the lusterless sun ablaze with the Prince of Death's flames, inflicting the death ailment upon foes. The blade stands as a relic of the cult that formed around Godwyn's half-dead state, worshipping the eclipse as an emblem of death's new foothold in the world."
                        , "Castle Sol, Mountaintops of the Giants"),
                Item.of("marikas_hammer",               "Marika's Hammer",                   ItemCategory.WEAPON,
                        "A stone hammer crafted in the Numen lands outside the Lands Between — the very tool with which Queen Marika shattered the Elden Ring and with which Radagon later attempted to repair it. The hammer partially broke upon the shattering, becoming splintered with rune fragments. That both acts — destruction and desperate repair — were performed with the same instrument speaks to the contradictory nature of Marika and Radagon as two halves of one being, one yielding to grief, the other straining against it."
                        , "Crumbling Farum Azula"),
                Item.of("godfrey_icon",                 "Godfrey Icon",                      ItemCategory.TALISMAN,
                        "A legendary talisman depicting the Elden Lord Godfrey. Godfrey was a ferocious warrior who, when he vowed to become a lord, took the Beast Regent Serosh upon his back to suppress the ceaseless lust for battle that raged within. The talisman memorializes the first Elden Lord — Marika's original consort — whose lineage would become the golden bloodline, and whose eventual exile from the Lands Between left the demigod hierarchy without its stabilizing patriarch, a vacuum that would deepen the chaos of the Shattering."
                        , "Forbidden Lands"),
                Item.of("marikas_soreseal",             "Marika's Soreseal",                 ItemCategory.TALISMAN,
                        "A legendary talisman in the form of an eye engraved with an Elden Rune, said to be the seal of Queen Marika. It greatly raises the wearer's mind, intelligence, faith, and arcane, but increases damage taken by a comparable measure. Solemn duty weighs upon the one beholden — not unlike a gnawing curse from which there is no deliverance. Found deep within Miquella's Haligtree, it suggests that the divine burden Marika carried — sustaining the Golden Order, containing death, holding the Elden Ring — was itself a kind of affliction. The seal implies a cost inseparable from divine authority."
                        , "Fort Faroth, Caelid"),
                Item.of("triple_rings_of_light",        "Triple Rings of Light",             ItemCategory.INCANTATION,
                        "One of the incantations of the Golden Order fundamentalists, this spell produces three rings of holy light and fires them outward before they arc back toward the caster. It was a gift from the young Miquella to his father, Radagon. This detail places Miquella squarely within the fundamentalist tradition before his disillusionment — at the time, he still believed the Golden Order could be perfected. His later abandonment of fundamentalism, when it failed to treat Malenia's rot or reverse Godwyn's fate, stands in direct contrast to this early act of filial devotion."
                ),
                Item.of("orders_blade",                 "Order's Blade",                     ItemCategory.INCANTATION,
                        "One of the incantations of the Golden Order fundamentalists, used by hunters of Those Who Live in Death. It enchants the caster's weapon with holy power especially damaging to the undead, and any felled by it cannot be revived. The role of such hunters is to stamp out defiled reason — all for the perfection of the Golden Order. The existence of this incantation is itself a consequence of Godwyn's half-death: once his corrupted body began seeding undeath through the Erdtree's roots, the Golden Order was forced to institutionalize the persecution of Those Who Live in Death, treating the symptom while refusing to address the cause."
                ),
                Item.of("unalloyed_gold_needle",        "Unalloyed Gold Needle",             ItemCategory.KEY_ITEM,
                        "Miquella crafted needles of unalloyed gold capable of warding away the meddling of outer gods and forestalling Malenia's incurable rot. Unalloyed gold — gold untouched by the influence of any outer god, including the Greater Will — was Miquella's answer to a theological problem: the Golden Order's god-endorsed framework could not cure ailments caused by other gods. These needles represent Miquella's pivot away from fundamentalism toward an independent, syncretic project of healing, and hint at his broader ambition to build an order free from divine coercion — the seed of the Haligtree."
                )
        );
    }

    public static boolean isPopulated() {
        return items().stream().noneMatch(i -> TODO.equals(i.description()));
    }
}
