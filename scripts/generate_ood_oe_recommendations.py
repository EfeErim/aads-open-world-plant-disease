#!/usr/bin/env python3
"""
Generate comprehensive OOD/OE dataset optimization recommendations.
Analyzes post-cleanup state and provides actionable improvements.
"""

import json
from pathlib import Path
from collections import defaultdict

def generate_recommendations() -> dict:
    """Generate comprehensive optimization recommendations."""

    root = Path("d:\\bitirme projesi")

    # Load cleanup log
    cleanup_log_path = root / "outputs" / "duplicate_removal_log.json"
    cleanup_log = {}
    if cleanup_log_path.exists():
        with open(cleanup_log_path) as f:
            cleanup_log = json.load(f)

    recommendations = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "cleanup_summary": {
            "duplicates_removed": cleanup_log.get("removed_count", 0),
            "space_freed_mb": round(cleanup_log.get("freed_space_mb", 0), 1),
            "failed_removals": cleanup_log.get("failed_count", 0)
        },
        "optimization_areas": [],
        "best_practices": [],
        "maintenance_schedule": []
    }

    # 1. Archive consolidation recommendations
    ood_archives = list((root / "data" / "ood_dataset").glob("*.zip"))
    if ood_archives:
        recommendations["optimization_areas"].append({
            "area": "Archive Consolidation",
            "priority": "MEDIUM",
            "current_state": f"{len(ood_archives)} archive files present",
            "recommendation": "Archives have been successfully extracted to 'final/' folders. Consider archiving original .zip files to cold storage (e.g., project backup drive) to reduce active storage.",
            "files": [str(f.name) for f in ood_archives],
            "expected_space_savings_mb": sum(f.stat().st_size for f in ood_archives) / (1024 * 1024)
        })

    # 2. Slice distribution analysis
    oe_root = root / "data" / "oe_dataset"
    slice_stats = defaultdict(lambda: defaultdict(int))

    for target_folder in oe_root.iterdir():
        if not target_folder.is_dir() or target_folder.name.startswith("_"):
            continue

        for image_file in target_folder.rglob("*.jpg"):
            slice_name = image_file.parent.name
            slice_stats[target_folder.name][slice_name] += 1

    imbalanced = []
    for target, slices in slice_stats.items():
        if len(slices) > 1:
            counts = list(slices.values())
            ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if ratio > 1.5:
                imbalanced.append({
                    "target": target,
                    "ratio": round(ratio, 2),
                    "slices": dict(slices)
                })

    if imbalanced:
        recommendations["optimization_areas"].append({
            "area": "Slice Distribution Balance",
            "priority": "LOW",
            "current_state": f"{len(imbalanced)} targets with imbalanced slices",
            "recommendation": "Consider data augmentation or re-collection for underrepresented slices to improve model robustness across all OE categories.",
            "targets_with_imbalance": imbalanced
        })

    # 3. Metadata recommendations
    recommendations["optimization_areas"].append({
        "area": "Metadata & Tracking",
        "priority": "MEDIUM",
        "recommendation": "Create _dataset_manifest.json files for each OOD/OE folder documenting source, collection date, quality checks, and version history for traceability.",
        "action_items": [
            "Add source provenance tracking",
            "Document collection methodology per slice",
            "Add quality control checkpoints",
            "Version dataset snapshots used in training runs"
        ]
    })

    # 4. Best practices
    recommendations["best_practices"] = [
        {
            "practice": "Hash verification",
            "description": "Maintain hash index of all images for future integrity checks and cross-dataset duplicate detection"
        },
        {
            "practice": "Incremental updates",
            "description": "Use modular slice folders to easily add new OOD/OE data without affecting existing training runs"
        },
        {
            "practice": "Backup strategy",
            "description": "Archive all OOD/OE source materials; keep extracted 'final/' and OE working folders for active training"
        },
        {
            "practice": "Version tracking",
            "description": "Include dataset version hash in training run metadata for reproducibility"
        }
    ]

    # 5. Maintenance schedule
    recommendations["maintenance_schedule"] = [
        {
            "frequency": "Weekly during active training",
            "task": "Monitor OOD FPR trends across new runs",
            "action": "Check production_readiness.json for ood_false_positive_rate"
        },
        {
            "frequency": "Monthly",
            "task": "Audit new image additions for duplicates",
            "action": "Run optimize_ood_oe_datasets.py on new data"
        },
        {
            "frequency": "Quarterly",
            "task": "Review slice distributions and coverage gaps",
            "action": "Analyze failure modes from validation runs"
        },
        {
            "frequency": "Annually",
            "task": "Deep archive consolidation and cleanup",
            "action": "Archive old versions, update documentation"
        }
    ]

    return recommendations

def main():
    root = Path("d:\\bitirme projesi")

    print("📊 Generating optimization recommendations...")
    recommendations = generate_recommendations()

    # Save recommendations
    output_path = root / "outputs" / "ood_oe_optimization_recommendations.json"
    with open(output_path, "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n✅ Recommendations saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("OOD/OE OPTIMIZATION SUMMARY")
    print("="*60)

    summary = recommendations["cleanup_summary"]
    print(f"\n🧹 Cleanup Results:")
    print(f"  Duplicates removed: {summary['duplicates_removed']:,}")
    print(f"  Space freed: {summary['space_freed_mb']:.1f} MB")
    print(f"  Errors: {summary['failed_removals']}")

    print(f"\n📋 Optimization Areas ({len(recommendations['optimization_areas'])}):")
    for item in recommendations["optimization_areas"]:
        print(f"  [{item['priority']}] {item['area']}")
        print(f"      {item['recommendation'][:70]}...")

    print(f"\n✨ Next Steps:")
    print(f"  1. Archive original .zip files to backup storage")
    print(f"  2. Add metadata manifests to OOD/OE folders")
    print(f"  3. Monitor OOD/FPR metrics in training runs")
    print(f"  4. Schedule quarterly slice distribution audits")

    return recommendations

if __name__ == "__main__":
    main()
