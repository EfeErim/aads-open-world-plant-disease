#!/usr/bin/env python3
"""Combine and reorganize OOD datasets into standard structure."""

import shutil
from collections import defaultdict
from pathlib import Path

OOD_BASE = Path(__file__).parent.parent / "data" / "ood_dataset"
DEST_DIR = OOD_BASE / "ood_tomato_fruit_combined"

# Category mappings: source_subfolder -> destination_category
# Pattern-based matching for flexible folder naming
SOURCE_MAPPINGS = {
    "domates_outlier": {
        "*": "unsupported_same_crop"  # all subfolders (disease classes) go here
    },
    "ood_tomato_fruit_yeni": {
        "Unsupported_tomato_fruit_unknowns level 1": "unsupported_same_crop",
        "non_fruit_tomato_secondary level 3": "unsupported_same_crop",
        "tomato_fruit_failure_cases level 2": "blur_or_occlusion",
        "scene_context_leak_check level 5": "blur_or_occlusion",
        "off_crop_fruit_secondary level 4": "other_crops_optional",
        "non_plant_tiny level 6": "other_crops_optional",
    },
    "ood_tomato_fruit_eski": {
        "Unsupported_tomato_fruit_unknowns level 1": "unsupported_same_crop",
        "non_fruit_tomato_secondary level 3": "unsupported_same_crop",
        "tomato_fruit_failure_cases level 2": "blur_or_occlusion",
        "scene_context_leak_check level 5": "blur_or_occlusion",
        "off_crop_fruit_secondary level 4": "other_crops_optional",
        "non_plant_tiny level 6": "other_crops_optional",
    }
}

CATEGORIES = ["unsupported_same_crop", "blur_or_occlusion", "other_crops_optional"]

def initialize_dest():
    """Create destination directory structure."""
    for cat in CATEGORIES:
        cat_path = DEST_DIR / cat
        cat_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Initialized {DEST_DIR}")

def count_files(path):
    """Count all files recursively."""
    return sum(1 for _ in path.rglob("*") if _.is_file())

def combine_datasets():
    """Combine all source OOD datasets into standard structure."""
    stats = defaultdict(int)

    for source_name, category_map in SOURCE_MAPPINGS.items():
        source_path = OOD_BASE / source_name
        if not source_path.exists():
            print(f"⚠ {source_name} not found, skipping")
            continue

        print(f"\nProcessing {source_name}...")

        # Get all subfolders in source
        for subfolder in sorted(source_path.iterdir()):
            if not subfolder.is_dir():
                continue

            # Determine destination category
            if "*" in category_map:
                dest_cat = category_map["*"]
            else:
                dest_cat = category_map.get(subfolder.name)
                if not dest_cat:
                    print(f"  ⚠ No mapping for {subfolder.name}, skipping")
                    continue

            # Copy all files to destination
            dest_path = DEST_DIR / dest_cat
            count = 0
            for file_path in subfolder.rglob("*"):
                if file_path.is_file():
                    try:
                        dest_file = dest_path / file_path.name
                        # Handle duplicates by adding source folder prefix
                        if dest_file.exists():
                            stem = file_path.stem
                            suffix = file_path.suffix
                            new_name = f"{stem}_{source_name}_{subfolder.name}{suffix}"
                            dest_file = dest_path / new_name

                        shutil.copy2(file_path, dest_file)
                        count += 1
                        stats[dest_cat] += 1
                    except Exception as e:
                        print(f"  ✗ Error copying {file_path}: {e}")

            print(f"  {subfolder.name:45} -> {dest_cat:25} ({count:,} files)")

    return stats

def print_summary(stats):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("COMBINED OOD DATASET STRUCTURE")
    print("=" * 70)

    total = 0
    for cat in CATEGORIES:
        count = stats[cat]
        total += count
        pct = 100.0 * count / total if total > 0 else 0
        print(f"{cat:30} : {count:6,} files ({pct:5.1f}%)")

    print("-" * 70)
    print(f"{'TOTAL':30} : {total:6,} files (100.0%)")
    print("=" * 70)

    # Verify files exist
    actual_total = count_files(DEST_DIR)
    print(f"\nDisk verification: {actual_total:,} files found")
    if actual_total == total:
        print("✓ Success: all files copied correctly")
    else:
        print(f"✗ Mismatch: expected {total}, found {actual_total}")

if __name__ == "__main__":
    print("Combining OOD datasets...\n")

    initialize_dest()
    stats = combine_datasets()
    print_summary(stats)
