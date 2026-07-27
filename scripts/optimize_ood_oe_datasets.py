#!/usr/bin/env python3
"""Analyze OOD/OE dataset folders for duplicates and simple cleanup opportunities."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for duplicate detection."""
    hash_obj = hashlib.new(algorithm)
    with filepath.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _empty_analysis(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "folders_analyzed": {},
        "total_files": 0,
        "duplicates_found": 0,
        "hash_to_files": defaultdict(list),
        "recommendations": [],
    }


def _record_duplicate_recommendations(analysis: dict[str, Any]) -> None:
    for file_hash, files in analysis["hash_to_files"].items():
        if len(files) <= 1:
            continue
        analysis["duplicates_found"] += len(files) - 1
        analysis["recommendations"].append(
            {
                "type": "duplicate_files",
                "hash": file_hash,
                "files": files,
                "keep": files[0],
                "remove": files[1:],
            }
        )


def analyze_ood_dataset(ood_root: Path) -> dict[str, Any]:
    """Analyze OOD dataset structure and find duplicates."""
    analysis = _empty_analysis(ood_root)

    final_dir = ood_root / "final"
    if not final_dir.exists():
        return analysis

    for image_file in final_dir.rglob("*.jpg"):
        analysis["total_files"] += 1
        try:
            file_hash = compute_file_hash(image_file)
            analysis["hash_to_files"][file_hash].append(str(image_file))
        except OSError as exc:
            print(f"Error hashing {image_file}: {exc}")

    _record_duplicate_recommendations(analysis)

    for folder in final_dir.iterdir():
        if not folder.is_dir():
            continue
        subfolders = {
            subfolder.name: len(list(subfolder.rglob("*.jpg")))
            for subfolder in folder.iterdir()
            if subfolder.is_dir()
        }
        analysis["folders_analyzed"][folder.name] = {
            "total_images": len(list(folder.rglob("*.jpg"))),
            "subfolders": subfolders,
        }

    archive_files = list(ood_root.glob("*.zip"))
    if archive_files:
        analysis["recommendations"].append(
            {
                "type": "archive_consolidation",
                "archives": [str(file) for file in archive_files],
                "note": "Consider consolidating extracted archives after validation.",
            }
        )

    return analysis


def analyze_oe_dataset(oe_root: Path) -> dict[str, Any]:
    """Analyze OE dataset structure and find duplicates."""
    analysis = _empty_analysis(oe_root)
    analysis["slice_distribution"] = {}

    if not oe_root.exists():
        return analysis

    for target_folder in oe_root.iterdir():
        if not target_folder.is_dir() or target_folder.name.startswith("_"):
            continue

        image_count = 0
        slices: defaultdict[str, int] = defaultdict(int)

        for image_file in target_folder.rglob("*.jpg"):
            analysis["total_files"] += 1
            image_count += 1
            slices[image_file.parent.name] += 1

            try:
                file_hash = compute_file_hash(image_file)
                analysis["hash_to_files"][file_hash].append(str(image_file))
            except OSError as exc:
                print(f"Error hashing {image_file}: {exc}")

        analysis["slice_distribution"][target_folder.name] = {
            "total_images": image_count,
            "slices": dict(slices),
        }

    _record_duplicate_recommendations(analysis)

    for target_name, data in analysis["slice_distribution"].items():
        slices = data["slices"]
        if len(slices) <= 1:
            continue
        slice_counts = list(slices.values())
        max_count = max(slice_counts)
        min_count = min(slice_counts)
        if min_count > 0 and max_count > min_count * 2:
            analysis["recommendations"].append(
                {
                    "type": "slice_imbalance",
                    "target": target_name,
                    "slices": slices,
                    "note": f"Imbalanced slices: {max_count / min_count:.1f}x spread.",
                }
            )

    return analysis


def generate_optimization_report(ood_analysis: dict[str, Any], oe_analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate comprehensive optimization report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "ood_analysis": ood_analysis,
        "oe_analysis": oe_analysis,
        "summary": {
            "ood_duplicates": ood_analysis["duplicates_found"],
            "oe_duplicates": oe_analysis["duplicates_found"],
            "total_ood_files": ood_analysis["total_files"],
            "total_oe_files": oe_analysis["total_files"],
            "optimization_opportunities": len(ood_analysis["recommendations"]) + len(oe_analysis["recommendations"]),
        },
        "action_plan": [],
    }

    if ood_analysis["duplicates_found"] > 0:
        report["action_plan"].append(
            {
                "priority": "HIGH",
                "action": "Remove OOD duplicates",
                "count": ood_analysis["duplicates_found"],
                "estimated_space_saved_mb": ood_analysis["duplicates_found"] * 0.5,
            }
        )

    if oe_analysis["duplicates_found"] > 0:
        report["action_plan"].append(
            {
                "priority": "HIGH",
                "action": "Remove OE duplicates",
                "count": oe_analysis["duplicates_found"],
                "estimated_space_saved_mb": oe_analysis["duplicates_found"] * 0.5,
            }
        )

    if any(item.get("type") == "archive_consolidation" for item in ood_analysis["recommendations"]):
        report["action_plan"].append(
            {
                "priority": "MEDIUM",
                "action": "Consolidate/cleanup OOD archives after validation",
                "note": "Archives should be retained for reproducibility but can be archived.",
            }
        )

    return report


def main(root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()

    print("Analyzing OOD dataset...")
    ood_analysis = analyze_ood_dataset(root / "data" / "ood_dataset")

    print("Analyzing OE dataset...")
    oe_analysis = analyze_oe_dataset(root / "data" / "oe_dataset")

    print("Generating optimization report...")
    report = generate_optimization_report(ood_analysis, oe_analysis)

    output_path = root / "outputs" / "ood_oe_optimization_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nReport saved to {output_path}")
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print("\nOOD Dataset:")
    print(f"  Total files: {ood_analysis['total_files']}")
    print(f"  Duplicates found: {ood_analysis['duplicates_found']}")
    print(f"  Folders: {len(ood_analysis['folders_analyzed'])}")

    print("\nOE Dataset:")
    print(f"  Total files: {oe_analysis['total_files']}")
    print(f"  Duplicates found: {oe_analysis['duplicates_found']}")
    print(f"  Targets: {len(oe_analysis['slice_distribution'])}")

    print(f"\nAction Items: {report['summary']['optimization_opportunities']}")
    for item in report["action_plan"]:
        print(f"  [{item['priority']}] {item['action']}")

    return report


if __name__ == "__main__":
    main()
