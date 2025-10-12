import os
import re
from collections import Counter

def clean_word(word):
    """Remove possessives and other noise from words."""
    return re.sub(r"\'s$", "", word).lower()

def main():
    """
    Main function to process all spelling datasets and create a clean frequency dictionary.
    """
    base_dir = "data/spelling"
    output_file = "frequency_dictionary_clean.txt"
    word_counts = Counter()

    print("Starting spelling dataset consolidation (excluding Google Ngram)...")

    # 1. Process Birkbeck (missp.dat)
    try:
        with open(os.path.join(base_dir, "missp.dat"), "r", encoding="utf-8") as f:
            for line in f:
                if "$" in line:
                    word = line.split("$")[1].strip()
                    word_counts[clean_word(word)] += 1
        print(f"✓ Processed Birkbeck dataset.")
    except FileNotFoundError:
        print(f"✗ WARNING: missp.dat not found. Skipping.")

    # 2. Process SCOWL (scowl.txt)
    try:
        with open(os.path.join(base_dir, "scowl.txt"), "r", encoding="utf-8") as f:
            for line in f:
                word_counts[clean_word(line.strip())] += 1
        print(f"✓ Processed SCOWL dataset.")
    except FileNotFoundError:
        print(f"✗ WARNING: scowl.txt not found. Skipping.")

    # 3. Process Google 10k (google-10000-english-no-swears.txt)
    try:
        with open(os.path.join(base_dir, "google-10000-english-no-swears.txt"), "r", encoding="utf-8") as f:
            for line in f:
                word_counts[clean_word(line.strip())] += 1
        print(f"✓ Processed Google 10k dataset.")
    except FileNotFoundError:
        print(f"✗ WARNING: google-10000-english-no-swears.txt not found. Skipping.")
        
    # 4. Process Wikipedia data
    try:
        with open(os.path.join(base_dir, "wikipedia.dat"), "r", encoding="utf-8") as f:
            for line in f:
                word_counts[clean_word(line.strip())] += 1
        print(f"✓ Processed Wikipedia dataset.")
    except FileNotFoundError:
        print(f"✗ WARNING: wikipedia.dat not found. Skipping.")
        
    # 5. Process Aspell data
    try:
        with open(os.path.join(base_dir, "aspell.dat"), "r", encoding="utf-8") as f:
            for line in f:
                word_counts[clean_word(line.strip())] += 1
        print(f"✓ Processed Aspell dataset.")
    except FileNotFoundError:
        print(f"✗ WARNING: aspell.dat not found. Skipping.")

    # Write the combined dictionary to the output file
    if not word_counts:
        print("No data processed, dictionary will be empty.")
        return
        
    print(f"\nTotal unique words consolidated: {len(word_counts)}")
    print(f"Writing to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for word, count in word_counts.most_common():
            # All words get a base frequency, as we are not using a frequency source
            f.write(f"{word} {count}\n")

    print(f"✓ Successfully created {output_file}")

if __name__ == "__main__":
    main() 