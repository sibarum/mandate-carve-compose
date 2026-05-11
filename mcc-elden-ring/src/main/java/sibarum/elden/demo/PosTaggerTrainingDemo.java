package sibarum.elden.demo;

import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;
import sibarum.elden.pos.ConlluParser;
import sibarum.elden.pos.ConlluParser.Sentence;
import sibarum.elden.pos.PosTrainer;
import sibarum.elden.pos.PosTrainer.TrainedPosLayer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Trains the POS layer on Universal Dependencies English EWT, then runs the
 * trained tagger on a sample Elden Ring sentence to show that the layer
 * correctly identifies function words and proper nouns — the failure modes
 * the binary span tagger overfit on.
 *
 * Usage:
 *   PosTaggerTrainingDemo <path-to-en_ewt-ud-train.conllu>
 *
 * Download the UD English EWT corpus from:
 *   https://github.com/UniversalDependencies/UD_English-EWT
 *
 * The file is CC BY-SA 4.0; not bundled.
 */
public final class PosTaggerTrainingDemo {

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 1;
    private static final int HIDDEN_DIM = 64;
    private static final long SEED = 42L;
    private static final int EPOCHS = 5;
    private static final double LR = 0.01;

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: PosTaggerTrainingDemo <path-to-en_ewt-ud-train.conllu>");
            System.err.println("Download from: https://github.com/UniversalDependencies/UD_English-EWT");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        System.out.println("Loading UD English EWT from " + conllu + " …");
        List<Sentence> sentences = ConlluParser.parse(conllu);
        System.out.println("Parsed " + sentences.size() + " sentences.");
        System.out.println();

        // Combined vocab: UD tokens + Elden Ring tokens (so the trained tagger
        // has embeddings for our downstream test prose even though only UD
        // tokens get supervised training).
        var extraVocab = CorpusVocabulary.tokens();
        System.out.println("Adding " + extraVocab.size() + " Elden Ring tokens to embedding vocab.");
        System.out.println();

        TrainedPosLayer pos = PosTrainer.train(
                sentences, extraVocab,
                EMBED_DIM, WINDOW_RADIUS, HIDDEN_DIM, SEED, EPOCHS, LR);
        System.out.println();

        // Inference on a sample Elden Ring sentence.
        String[] samples = {
                "Marika shattered the Elden Ring after Godwyn's death.",
                "Ranni gave the dark moon ring to her chosen consort.",
                "The Rune of Death was stolen from Maliketh by the Black Knives."
        };

        for (String sample : samples) {
            System.out.println("================================================================");
            System.out.println("input:  \"" + sample + "\"");
            System.out.println("================================================================");
            List<OffsetToken> toks = OffsetTokenizer.tokenize(sample);
            List<String> forms = toks.stream().map(OffsetToken::text).collect(Collectors.toList());
            System.out.println("predicted POS tags:");
            for (int i = 0; i < forms.size(); i++) {
                String tag = pos.predict(forms, i);
                System.out.printf("  %-15s  %s%n", "'" + forms.get(i) + "'", tag);
            }
            System.out.println();
        }
    }
}
