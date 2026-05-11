package sibarum.elden.demo;

import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.ItemCategory;
import sibarum.elden.corpus.ShatteringEra;


import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public final class CorpusDemo {

    public static void main(String[] args) {
        List<Item> items = ShatteringEra.items();

        System.out.println("Shattering-era corpus");
        System.out.println("=====================");
        System.out.println("total items: " + items.size());
        System.out.println();

        Map<ItemCategory, Long> byCategory = new TreeMap<>();
        for (Item item : items) {
            byCategory.merge(item.category(), 1L, Long::sum);
        }
        System.out.println("by category:");
        byCategory.forEach((cat, count) -> System.out.printf("  %-12s %d%n", cat, count));
        System.out.println();

        System.out.println("items:");
        for (Item item : items) {
            System.out.printf("  [%s] %s%n", item.category(), item.name());
            System.out.println("    id:       " + item.id());
            item.location().ifPresent(loc -> System.out.println("    location: " + loc));
            System.out.println("    text:     " + truncate(item.description(), 80));
            System.out.println();
        }

        System.out.println("corpus populated: " + ShatteringEra.isPopulated());
    }

    private static String truncate(String s, int max) {
        if (s.length() <= max) return s;
        return s.substring(0, max - 3) + "...";
    }
}
