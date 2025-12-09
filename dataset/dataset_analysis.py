#!/usr/bin/env python3
import json
import csv
import argparse
from collections import defaultdict

ALLOWED_CATEGORIES = {
    "health and mental well-being",
    "relationships and sentiments",
    "identity and diversity",
    "sexual",
    "religion and philosophy",
    "security",
    "politics and society",
}

def normalize_cat(c: str) -> str:
    """Normalize a category string for matching."""
    if c is None:
        return ""
    c = c.strip().lower()
    c = " ".join(c.split())
    return c

def extract_categories(cat_field):
    """
    Handle both formats:
    - string: "a/b/c"
    - list: ["a", "b", "c"]
    Returns a list of normalized categories filtered by ALLOWED_CATEGORIES.
    """
    if cat_field is None:
        return []

    cats = []

    if isinstance(cat_field, str):
        raw_parts = [p for p in cat_field.split("/") if p.strip()]
    elif isinstance(cat_field, list):
        raw_parts = cat_field
    else:
        raw_parts = []

    for p in raw_parts:
        norm = normalize_cat(p)
        if norm in ALLOWED_CATEGORIES and norm not in cats:
            cats.append(norm)

    return cats

def analyze_dataset(input_path: str, output_csv: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    n_sensitive = 0
    n_non_sensitive = 0

    # category -> stats
    cat_stats = {
        c: {
            "total": 0,
            "sensitive": 0,
            "non_sensitive": 0,
            "example_question": None,
        }
        for c in ALLOWED_CATEGORIES
    }

    for item in data:
        total += 1

        question = item.get("question_en", "").strip()
        label = item.get("sensitive?", item.get("sensitive", None))

        # Normalize label to 0/1
        if label in [True, "1", "true", "True"]:
            label = 1
        elif label in [False, "0", "false", "False"]:
            label = 0

        try:
            label = int(label)
        except (TypeError, ValueError):
            # If label is missing or invalid, skip stats on sensitivity
            label = None

        if label == 1:
            n_sensitive += 1
        elif label == 0:
            n_non_sensitive += 1

        cats = extract_categories(item.get("category"))

        # Update per-category stats
        for c in cats:
            cat_stats[c]["total"] += 1
            if label == 1:
                cat_stats[c]["sensitive"] += 1
            elif label == 0:
                cat_stats[c]["non_sensitive"] += 1

            # Store first example question if not already set
            if not cat_stats[c]["example_question"] and question:
                cat_stats[c]["example_question"] = question

    # Print a quick summary on screen
    print(f"Total prompts: {total}")
    print(f"Sensitive: {n_sensitive}")
    print(f"Non-sensitive: {n_non_sensitive}")
    print()
    print("Per-category stats:")
    for c in ALLOWED_CATEGORIES:
        stats = cat_stats[c]
        print(
            f"- {c}: total={stats['total']}, "
            f"sensitive={stats['sensitive']}, non_sensitive={stats['non_sensitive']}"
        )

    # Write CSV
    fieldnames = [
        "category",
        "total_prompts",
        "sensitive_prompts",
        "non_sensitive_prompts",
        "example_question",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in ALLOWED_CATEGORIES:
            stats = cat_stats[c]
            writer.writerow(
                {
                    "category": c,
                    "total_prompts": stats["total"],
                    "sensitive_prompts": stats["sensitive"],
                    "non_sensitive_prompts": stats["non_sensitive"],
                    "example_question": stats["example_question"] or "",
                }
            )

    print(f"\nCategory statistics written to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the SENSY dataset and export category statistics to CSV."
    )
    parser.add_argument(
        "-i", "--input",
        default="SENSY_clean.json",
        help="Input JSON file (default: SENSY_clean.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="sensy_category_stats.csv",
        help="Output CSV file (default: sensy_category_stats.csv)",
    )
    args = parser.parse_args()

    analyze_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
