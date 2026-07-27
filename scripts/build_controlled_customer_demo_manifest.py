"""Build the controlled customer-demo manifest from the latest M2 stress result."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULT_JSON = Path("docs/demo_results/m2/20260706T140135Z/m2_demo_checklist_run.json")
DEFAULT_SOURCE_MANIFEST = Path("docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv")
DEFAULT_OUTPUT = Path("docs/demo_assets/customer_demo_manifest/customer_demo_manifest.csv")
DEFAULT_SUMMARY_JSON = Path("docs/demo_assets/customer_demo_manifest/customer_demo_manifest_summary.json")
DEFAULT_SUMMARY_MD = Path("docs/demo_assets/customer_demo_manifest/customer_demo_manifest_summary.md")

EXCLUDED_STRESS_ROWS = {"demo_236", "demo_432"}

SUPPORT_LABELS = {
    "apricot__leaf": "strong",
    "strawberry__leaf": "strong",
    "grape__leaf": "strong",
    "tomato__leaf": "caution",
    "tomato__fruit": "caution",
    "apricot__fruit": "caution",
    "grape__fruit": "caution",
    "strawberry__fruit": "problematic",
    "unknown_crop": "safety_review",
    "non_plant": "safety_review",
    "tomato__unknown_part": "safety_review",
    "grape__unknown_part": "safety_review",
}

TARGET_SELECTION_COUNTS = {
    "apricot__leaf": 6,
    "strawberry__leaf": 6,
    "grape__leaf": 6,
    "tomato__leaf": 4,
    "tomato__fruit": 4,
    "apricot__fruit": 4,
    "grape__fruit": 4,
    "strawberry__fruit": 2,
    "unknown_crop": 4,
    "non_plant": 4,
    "tomato__unknown_part": 2,
    "grape__unknown_part": 2,
}

BASE_FIELDS = [
    "image_id",
    "source",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "expected_behavior",
    "notes",
    "original_source",
    "resolved_source_path",
    "disease_class",
    "user_like_bucket",
]

EXTRA_FIELDS = [
    "support_label",
    "demo_bucket",
    "selection_reason",
    "latest_result_status",
    "latest_result_pass_fail",
    "latest_predicted_crop",
    "latest_predicted_part",
    "latest_predicted_disease",
    "latest_failure_bucket",
]


@dataclass(frozen=True)
class BuildResult:
    rows: list[dict[str, str]]
    summary: dict[str, Any]


def _read_csv_by_image_id(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = {str(row["image_id"]): dict(row) for row in reader}
        return rows, list(reader.fieldnames or [])


def _read_result_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a list-valued 'rows' field")
    return rows, payload


def _bucket_for_target(target: str) -> str:
    label = SUPPORT_LABELS[target]
    if label == "strong":
        return "answer_strong"
    if label == "caution":
        return "answer_caution"
    if label == "problematic":
        return "answer_problematic_limited"
    return "safety_review"


def _selection_reason(target: str) -> str:
    label = SUPPORT_LABELS[target]
    if label == "strong":
        return "20260706T140135Z strong target; pass row selected for customer acceptance"
    if label == "caution":
        return "20260706T140135Z caution target; pass row selected with limitation label"
    if label == "problematic":
        return "20260706T140135Z problematic target; minimal pass row kept for limitation-aware demo"
    return "20260706T140135Z safety/review probe; pass row selected to show abstain/review behavior"


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    notes = str(row.get("notes") or "")
    image_id = str(row.get("image_id") or "")
    bucket_order = {"bucket=clear": 0, "bucket=phone": 1, "bucket=user_like": 1, "bucket=field": 1}
    bucket_rank = 5
    for marker, rank in bucket_order.items():
        if marker in notes:
            bucket_rank = min(bucket_rank, rank)
    source_rank = 0 if "source_split=test" in notes else 1
    return bucket_rank, source_rank, image_id


def build_controlled_manifest(result_json: Path, source_manifest: Path) -> BuildResult:
    source_rows, source_fields = _read_csv_by_image_id(source_manifest)
    result_rows, result_payload = _read_result_rows(result_json)
    result_by_id = {str(row["image_id"]): row for row in result_rows}

    selected_rows: list[dict[str, str]] = []
    missing_targets: dict[str, dict[str, int]] = {}

    for target, target_count in TARGET_SELECTION_COUNTS.items():
        candidates = [
            row
            for row in result_rows
            if row.get("expected_target") == target
            and row.get("pass_fail") == "pass"
            and row.get("image_id") not in EXCLUDED_STRESS_ROWS
            and row.get("image_id") in source_rows
        ]
        candidates.sort(key=_candidate_sort_key)
        chosen = candidates[:target_count]
        if len(chosen) < target_count:
            missing_targets[target] = {"needed": target_count, "available": len(chosen)}
            continue

        for result_row in chosen:
            image_id = str(result_row["image_id"])
            source_row = dict(source_rows[image_id])
            support_label = SUPPORT_LABELS[target]
            demo_bucket = _bucket_for_target(target)
            notes = source_row.get("notes", "")
            suffix = f"controlled_demo; support_label={support_label}; demo_bucket={demo_bucket}"
            source_row["notes"] = f"{notes}; {suffix}" if notes else suffix
            source_row.update(
                {
                    "support_label": support_label,
                    "demo_bucket": demo_bucket,
                    "selection_reason": _selection_reason(target),
                    "latest_result_status": str(result_row.get("actual_status") or ""),
                    "latest_result_pass_fail": str(result_row.get("pass_fail") or ""),
                    "latest_predicted_crop": str(result_row.get("predicted_crop") or ""),
                    "latest_predicted_part": str(result_row.get("predicted_part") or ""),
                    "latest_predicted_disease": str(result_row.get("predicted_disease") or ""),
                    "latest_failure_bucket": str(result_row.get("failure_bucket") or ""),
                }
            )
            selected_rows.append(source_row)

    if missing_targets:
        raise RuntimeError(f"Not enough passing rows for controlled demo manifest: {missing_targets}")

    fieldnames = [field for field in BASE_FIELDS if field in source_fields]
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    support_counts = Counter(row["support_label"] for row in selected_rows)
    bucket_counts = Counter(row["demo_bucket"] for row in selected_rows)
    target_counts = Counter(row["expected_target"] for row in selected_rows)
    excluded_present = sorted(image_id for image_id in EXCLUDED_STRESS_ROWS if image_id in result_by_id)

    summary = {
        "schema_version": "v1_controlled_customer_demo_manifest",
        "source_result": str(result_json),
        "source_manifest": str(source_manifest),
        "source_result_finished_at": result_payload.get("finished_at"),
        "total_rows": len(selected_rows),
        "support_label_counts": dict(sorted(support_counts.items())),
        "demo_bucket_counts": dict(sorted(bucket_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "excluded_stress_rows": excluded_present,
        "fieldnames": fieldnames,
    }
    return BuildResult(rows=selected_rows, summary=summary)


def write_outputs(build: BuildResult, output: Path, summary_json: Path, summary_md: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = build.summary["fieldnames"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(build.rows)

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump({**build.summary, "output_manifest": str(output)}, file, ensure_ascii=False, indent=2)
        file.write("\n")

    lines = [
        "# Controlled Customer Demo Manifest",
        "",
        "Generated from `docs/demo_results/m2/20260706T140135Z/m2_demo_checklist_run.json`.",
        "",
        "## Support Labels",
        "",
        "| Label | Targets | Demo meaning |",
        "|---|---|---|",
        "| `strong` | `apricot__leaf`, `strawberry__leaf`, `grape__leaf` | Primary supported demo surfaces. |",
        "| `caution` | `tomato__leaf`, `tomato__fruit`, `apricot__fruit`, `grape__fruit` | Use with explicit limitation language. |",
        "| `problematic` | `strawberry__fruit` | Keep minimal passing examples; present remaining stress cases as limitations. |",
        "| `safety_review` | `unknown_crop`, `non_plant`, `tomato__unknown_part`, `grape__unknown_part` | Demonstrates review/abstain behavior for unsupported inputs. |",
        "",
        "## Selected Counts",
        "",
        "| Expected target | Rows |",
        "|---|---:|",
    ]
    for target, count in build.summary["target_counts"].items():
        lines.append(f"| `{target}` | {count} |")
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "The controlled acceptance manifest excludes all latest-result failure rows and explicitly excludes stress regressions `demo_236` and `demo_432`.",
            "",
            "## Commands",
            "",
            "```powershell",
            ".\\scripts\\python.cmd scripts\\build_controlled_customer_demo_manifest.py",
            ".\\scripts\\python.cmd scripts\\run_demo_checklist.py --mode asset-audit --no-checklist --extra-manifest docs\\demo_assets\\customer_demo_manifest\\customer_demo_manifest.csv --output .runtime_tmp\\customer_demo_asset_audit.json --markdown-output .runtime_tmp\\customer_demo_asset_audit.md",
            ".\\scripts\\python.cmd scripts\\run_demo_checklist.py --no-checklist --extra-manifest docs\\demo_assets\\customer_demo_manifest\\customer_demo_manifest.csv --device cuda --adapter-root runs --batch-size 4 --adapter-batch-size 2 --handoff-cache .runtime_tmp\\m2_customer_demo_handoff_cache.json",
            "```",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build = build_controlled_manifest(args.result_json, args.source_manifest)
    write_outputs(build, args.output, args.summary_json, args.summary_md)
    print(f"wrote {args.output} ({build.summary['total_rows']} rows)")
    print(f"wrote {args.summary_json}")
    print(f"wrote {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
