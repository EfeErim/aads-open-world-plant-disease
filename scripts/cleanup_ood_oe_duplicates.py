#!/usr/bin/env python3
"""
Remove duplicate files from OOD and OE datasets based on analysis report.
Safely removes duplicates while preserving one copy of each unique file.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def remove_duplicates_from_report(report_path: Path, dry_run: bool = True) -> dict:
    """Remove duplicates based on optimization report."""

    with open(report_path) as f:
        report = json.load(f)

    removed_count = 0
    failed_count = 0
    freed_space_bytes = 0
    removal_log = []

    # Process OOD recommendations
    for rec in report.get("ood_analysis", {}).get("recommendations", []):
        if rec.get("type") == "duplicate_files":
            for dup_file in rec.get("remove", []):
                dup_path = Path(dup_file)
                try:
                    file_size = dup_path.stat().st_size if dup_path.exists() else 0

                    if dry_run:
                        removal_log.append({
                            "action": "would_remove",
                            "file": dup_file,
                            "size_bytes": file_size
                        })
                        freed_space_bytes += file_size
                        removed_count += 1
                    else:
                        if dup_path.exists():
                            dup_path.unlink()
                            removal_log.append({
                                "action": "removed",
                                "file": dup_file,
                                "size_bytes": file_size
                            })
                            freed_space_bytes += file_size
                            removed_count += 1
                except Exception as e:
                    removal_log.append({
                        "action": "error",
                        "file": dup_file,
                        "error": str(e)
                    })
                    failed_count += 1

    # Process OE recommendations
    for rec in report.get("oe_analysis", {}).get("recommendations", []):
        if rec.get("type") == "duplicate_files":
            for dup_file in rec.get("remove", []):
                dup_path = Path(dup_file)
                try:
                    file_size = dup_path.stat().st_size if dup_path.exists() else 0

                    if dry_run:
                        removal_log.append({
                            "action": "would_remove",
                            "file": dup_file,
                            "size_bytes": file_size
                        })
                        freed_space_bytes += file_size
                        removed_count += 1
                    else:
                        if dup_path.exists():
                            dup_path.unlink()
                            removal_log.append({
                                "action": "removed",
                                "file": dup_file,
                                "size_bytes": file_size
                            })
                            freed_space_bytes += file_size
                            removed_count += 1
                except Exception as e:
                    removal_log.append({
                        "action": "error",
                        "file": dup_file,
                        "error": str(e)
                    })
                    failed_count += 1

    return {
        "dry_run": dry_run,
        "timestamp": datetime.now().isoformat(),
        "removed_count": removed_count,
        "failed_count": failed_count,
        "freed_space_bytes": freed_space_bytes,
        "freed_space_mb": freed_space_bytes / (1024 * 1024),
        "log": removal_log
    }

def main():
    # Check for command line argument for auto-yes
    auto_yes = "--yes" in sys.argv or "--force" in sys.argv

    root = Path("d:\\bitirme projesi")
    report_path = root / "outputs" / "ood_oe_optimization_report.json"

    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        sys.exit(1)

    print("="*60)
    print("OOD & OE DUPLICATE REMOVAL")
    print("="*60)

    # First run dry-run
    print("\n📋 Running DRY-RUN analysis...")
    dry_result = remove_duplicates_from_report(report_path, dry_run=True)

    print(f"\n✓ Would remove: {dry_result['removed_count']} files")
    print(f"✓ Would free: {dry_result['freed_space_mb']:.1f} MB")
    print(f"✓ Errors: {dry_result['failed_count']}")

    # Ask for confirmation
    print("\n" + "="*60)
    if auto_yes:
        response = "yes"
        print("Auto-proceeding with removal (--yes flag set)...")
    else:
        response = input("Proceed with actual removal? (yes/no): ").strip().lower()

    if response == "yes":
        print("\n🔄 Executing duplicate removal...")
        result = remove_duplicates_from_report(report_path, dry_run=False)

        print(f"\n✅ Removed: {result['removed_count']} files")
        print(f"✅ Freed: {result['freed_space_mb']:.1f} MB")
        print(f"⚠️  Errors: {result['failed_count']}")

        # Save result log
        log_path = root / "outputs" / "duplicate_removal_log.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n📝 Removal log saved: {log_path}")
    else:
        print("\n⏭️  Skipping removal. Run again with 'yes' to proceed.")

if __name__ == "__main__":
    main()
