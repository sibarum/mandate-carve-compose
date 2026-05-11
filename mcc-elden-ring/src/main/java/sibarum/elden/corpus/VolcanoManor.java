package sibarum.elden.corpus;

import java.util.List;

public final class VolcanoManor {

    private VolcanoManor() {
    }

    public static List<Item> items() {
        return List.of(
                Item.of("volcano_manor_invitation", "Volcano Manor Invitation", ItemCategory.KEY_ITEM,
                        "Volcano Manor Invitation: A letter sealed in red wax bearing the crest of Tanith, lady "
                                + "of Volcano Manor. The invitation is extended to Tarnished of sufficient strength "
                                + "and discontent, offering them a place within a conspiracy that brews beneath Mt. "
                                + "Gelmir. Those who accept are bound into service as Recusants — assassins dispatched "
                                + "against champions of the Golden Order — in pursuit of a vengeance long nursed by "
                                + "the manor's true master, the once-Lord Rykard.",
                        "Mt. Gelmir"
                ),

                Item.of("ryas_necklace", "Rya's Necklace", ItemCategory.KEY_ITEM,
                        "Rya's Necklace: A pendant belonging to Rya, attendant and emissary of Lady Tanith, stolen "
                                + "from her by a thief at the Boilprawn Shack on the shore of Lake Liurnia. Recovering "
                                + "the necklace and returning it to Rya earns her gratitude and, with it, an "
                                + "introduction to her mistress at Volcano Manor — the first step by which a Tarnished "
                                + "is drawn into the conspiracy of the Recusants. The pendant itself is unremarkable "
                                + "in craft, but Rya treasures it as a token of the station Tanith has granted her, "
                                + "unaware that the form she wears and the life she remembers are themselves a gift "
                                + "of her lord's serpentine making.",
                        "Liurnia of the Lakes"
                ),

                Item.of("letter_from_volcano_manor", "Letter from Volcano Manor", ItemCategory.KEY_ITEM,
                        "Letter from Volcano Manor: A contract from Tanith naming a target whose death would advance "
                                + "the manor's cause. The targets are champions and exemplars of the Golden Order, "
                                + "and their elimination is framed as both retribution against the order that "
                                + "condemned Rykard and recruitment by demonstration — proof that the assassin shares "
                                + "the manor's grievance and is fit to stand among its Recusants.",
                        "Volcano Manor"
                ),

                Item.of("recusant_finger", "Recusant Finger", ItemCategory.KEY_ITEM,
                        "Recusant Finger: A severed finger used to invade the worlds of other Tarnished on behalf "
                                + "of Volcano Manor. The Recusants are the manor's covenant of invaders, sworn to "
                                + "carry out the assassinations contracted by Tanith and, more broadly, to wage a "
                                + "secret war against the chosen of the Greater Will wherever they may be found.",
                        "Volcano Manor"
                ),

                Item.of("serpent_hunter", "Serpent-Hunter", ItemCategory.WEAPON,
                        "Serpent-Hunter: A great spear forged specifically for the slaying of the God-Devouring "
                                + "Serpent, Rykard's monstrous new form. The weapon is left in the chamber outside "
                                + "his lair by an unknown ally — likely Recusant Bernahl in his final moment of "
                                + "conscience, or another who remembered Rykard as he once was — for use by any "
                                + "Tarnished who would put an end to the apostate Lord's blasphemous designs upon "
                                + "godhood through consumption.",
                        "Volcano Manor"
                ),

                Item.of("remembrance_blasphemous", "Remembrance of the Blasphemous", ItemCategory.REMEMBRANCE,
                        "Remembrance of the Blasphemous: The remembrance of Rykard, Lord of Blasphemy, second son "
                                + "of Queen Rennala and Radagon and once a demigod of the Golden Order. Rykard "
                                + "renounced the order entirely, allied himself with the Great Serpent that had long "
                                + "dwelt beneath Mt. Gelmir, and sought to become a god through the endless "
                                + "consumption of heroes — a path of apotheosis through devouring rather than "
                                + "through grace, in deliberate inversion of every tenet the Erdtree had imposed.",
                        "Volcano Manor"
                ),

                Item.of("rykards_rancor", "Rykard's Rancor", ItemCategory.SORCERY,
                        "Rykard's Rancor: A sorcery born of Rykard's hatred, manifesting as screaming heads that "
                                + "pursue their target with the fury of the consumed. The spell preserves the voices "
                                + "of those Rykard devoured in his pursuit of godhood, their rancor folded into his "
                                + "own and turned outward upon any who would still kneel before the Erdtree that "
                                + "cast him out.",
                        "Volcano Manor"
                ),

                Item.of("blasphemous_blade", "Blasphemous Blade", ItemCategory.WEAPON,
                        "Blasphemous Blade: A greatsword forged from Rykard's remembrance, bearing the face of the "
                                + "Fell God whose flame Rykard came to revere in his final apostasy. The blade "
                                + "restores its wielder upon the slaying of foes, mirroring the manner of Rykard's "
                                + "own attempted ascension — that godhood might be achieved not by the gift of an "
                                + "outer power but by the perpetual taking of life from others.",
                        "Volcano Manor"
                ),

                Item.of("magma_wyrm_scalesword", "Magma Wyrm's Scalesword", ItemCategory.WEAPON,
                        "Magma Wyrm's Scalesword: A curved greatsword wielded by the magma wyrms that nest within "
                                + "Mt. Gelmir and the lava-filled depths beneath Volcano Manor. These wyrms are "
                                + "descendants of the ancient dragons, twisted by their long dwelling in fire, and "
                                + "their presence beneath the manor is no accident — Rykard cultivated them as part "
                                + "of the infernal court he gathered around himself once he had abandoned the "
                                + "company of his fellow demigods.",
                        "Mt. Gelmir"
                ),

                Item.of("drawing_room_key", "Drawing-Room Key", ItemCategory.KEY_ITEM,
                        "Drawing-Room Key: A key granted to favored Recusants, opening the inner chambers of "
                                + "Volcano Manor where Tanith holds private audience. Beyond the drawing room lies "
                                + "the passage descending to Rykard's lair, and the trust implied by the key marks "
                                + "its bearer as one no longer merely contracted by the manor but counted, however "
                                + "provisionally, among its household."
                ),

                Item.of("iron_maiden_set", "Iron Maiden Set", ItemCategory.ARMOR,
                        "Iron Maiden Set: The armor worn by the Abductor Virgins — mechanical contraptions that "
                                + "patrol the slopes of Mt. Gelmir and the corridors beneath Volcano Manor, seizing "
                                + "wayfarers and delivering them down into the manor's depths. The captured are fed "
                                + "either to the magma wyrms or to Rykard himself, who in his serpent form requires "
                                + "an unceasing supply of heroes to consume.",
                        "Mt. Gelmir"
                ),

                Item.of("tonic_of_forgetfulness", "Tonic of Forgetfulness", ItemCategory.KEY_ITEM,
                        "Tonic of Forgetfulness: A draught provided by Tanith that, when administered, erases a "
                                + "specific memory from its drinker. Tanith offers the tonic to be given to Rya, her "
                                + "attendant and emissary, so that Rya might remain unaware of her own true nature "
                                + "as a serpent-woman of Rykard's brood. The gesture is presented as a mercy — that "
                                + "Rya may continue to live as the noblewoman she believes herself to be — though it "
                                + "is equally a means of keeping a useful servant bound to the manor in ignorance of "
                                + "what her lord has made of his household.",
                        "Volcano Manor"
                )
        );
    }
}
