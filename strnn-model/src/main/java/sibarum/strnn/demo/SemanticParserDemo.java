package sibarum.strnn.demo;

import sibarum.strnn.cache.semantic.SemExpr;
import sibarum.strnn.cache.semantic.SemRelation;
import sibarum.strnn.cache.semantic.SemanticParser;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Phase 0 of the KV-cache demo line: load the hand-crafted semantic ontology
 * from {@code resources/sample-semantics.txt}, parse it into typed records,
 * and report what the parser sees. No training, no embeddings — just verify
 * the file ingests cleanly and surface the structure for downstream phases.
 *
 * Diagnostics:
 *   1) Every non-comment, non-blank line parses without errors.
 *   2) Round-trip stability: parse → toString → re-parse produces an equal AST.
 *      (Small risk: redundant parens in the printed form change parse trees;
 *      the identity we assert is structural equality of the AST.)
 *   3) Vocabulary and operator-usage stats reported for downstream sizing.
 *   4) A handful of sample relations printed to confirm precedence parsed
 *      sensibly across qualifier / conjunction / union nesting.
 */
public final class SemanticParserDemo {

    public static void main(String[] args) throws IOException {
        String source = loadResource("/sample-semantics.txt");

        System.out.println("=== Diagnostic 1: parse all relations ===");
        List<SemRelation> relations = SemanticParser.parseAll(source);
        System.out.printf(Locale.ROOT, "  parsed %d relations from %d lines of source%n",
                relations.size(), source.split("\\R").length);
        require(!relations.isEmpty(), "no relations parsed from source");
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 2: round-trip parse → toString → re-parse ===");
        int rtMismatches = 0;
        for (SemRelation r : relations) {
            String printed = r.toString();
            SemRelation reparsed;
            try {
                reparsed = SemanticParser.parseLine(printed);
            } catch (RuntimeException e) {
                throw new AssertionError("round-trip parse failed for: " + printed, e);
            }
            if (!r.equals(reparsed)) {
                rtMismatches++;
                if (rtMismatches <= 3) {
                    System.out.printf("    MISMATCH:%n      orig:    %s%n      printed: %s%n      reparsed: %s%n",
                            r, printed, reparsed);
                }
            }
        }
        require(rtMismatches == 0, rtMismatches + " round-trip mismatches");
        System.out.printf(Locale.ROOT, "  round-trip stable across all %d relations%n", relations.size());
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 3: vocabulary and operator usage ===");
        Set<String> atoms = SemanticParser.collectAtoms(relations);
        Map<String, Integer> opCounts = countOperators(relations);
        System.out.printf(Locale.ROOT, "  unique atoms (vocabulary size): %d%n", atoms.size());
        System.out.printf(Locale.ROOT, "  dichotomy pairs              : %d%n", relations.size());
        for (var e : opCounts.entrySet()) {
            System.out.printf(Locale.ROOT, "  %-20s : %d%n", e.getKey(), e.getValue());
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 4: sample parses ===");
        // Pick a few characteristic lines covering different operator combinations.
        String[] interesting = {
                "(true | false) ~> (truth & logic & philosophy)",
                "(zero | infinity) ~> quantity",
                "(absence | presence) ~> abstract:presence",
                "(predator | prey) ~> (hunting & cause+effect)",
                "(sharp | dull) ~> (physical:property & intelligence:description)",
                "(ancestor | descendent) ~> (chronology & ancestry, cause+effect)",
                "(participant | observer) ~> (intention & disposition & action & social:role)"
        };
        for (String line : interesting) {
            SemRelation r = SemanticParser.parseLine(line);
            System.out.printf(Locale.ROOT, "  %s%n", r);
        }
        System.out.println("  PASS");

        System.out.printf(Locale.ROOT, "%nPhase 0: parser ingests %d relations covering %d atoms; "
                        + "round-trip stable; ready for embedding training.%n",
                relations.size(), atoms.size());
    }

    // -----------------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------------

    private static String loadResource(String path) throws IOException {
        try (InputStream in = SemanticParserDemo.class.getResourceAsStream(path)) {
            if (in == null) throw new IOException("resource not found: " + path);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static Map<String, Integer> countOperators(List<SemRelation> relations) {
        Map<String, Integer> c = new LinkedHashMap<>();
        c.put("union (&)", 0);
        c.put("qualifier (:)", 0);
        c.put("conjunction (+)", 0);
        c.put("composition (*)", 0);
        for (SemRelation r : relations) {
            countIn(r.rhs(), c);
        }
        return c;
    }

    private static void countIn(SemExpr e, Map<String, Integer> c) {
        switch (e) {
            case SemExpr.Atom a -> { /* leaf */ }
            case SemExpr.Qualified q -> {
                c.merge("qualifier (:)", 1, Integer::sum);
                countIn(q.head(), c);
                countIn(q.facet(), c);
            }
            case SemExpr.Composition cp -> {
                c.merge("composition (*)", 1, Integer::sum);
                countIn(cp.left(), c);
                countIn(cp.right(), c);
            }
            case SemExpr.Conjunction cj -> {
                c.merge("conjunction (+)", 1, Integer::sum);
                countIn(cj.left(), c);
                countIn(cj.right(), c);
            }
            case SemExpr.Union u -> {
                c.merge("union (&)", 1, Integer::sum);
                for (SemExpr m : u.members()) countIn(m, c);
            }
        }
    }

    private static void require(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }
}
