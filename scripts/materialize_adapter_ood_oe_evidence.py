#!/usr/bin/env python3
"""Materialize reviewed adapter OOD/OE evidence and emit the strict v2 combined manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapter_ood_oe_materialization import build_materialization_plan, materialize_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/internet_image_candidates/turkey_adapter_ood"))
    parser.add_argument("--evidence-root", type=Path, default=Path("data/adapter_ood_oe_evidence"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared_runtime_datasets"))
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        default=Path(
            "data/internet_image_candidates/turkey_adapter_ood/"
            "legacy_reviewed_adapter_ood_oe_manifest.csv"
        ),
        help="Immutable reviewed v1 placement snapshot upgraded into the replaceable v2 output.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/prepared_runtime_datasets/adapter_ood_oe_evidence_manifest.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(".runtime_tmp/adapter_ood_oe_materialization_summary.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-missing-source", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        plan = build_materialization_plan(
            repo_root=REPO_ROOT,
            manifest_root=_resolve(args.manifest_root),
            evidence_root=_resolve(args.evidence_root),
            prepared_root=_resolve(args.prepared_root),
            legacy_manifest_path=_resolve(args.legacy_manifest),
            fail_on_missing_source=bool(args.fail_on_missing_source),
        )
        result = {
            key: value
            for key, value in plan.items()
            if key not in {"operations", "manifest_rows"}
        }
        result["dry_run"] = bool(args.dry_run)
        result["ok"] = not bool(plan["missing_source_count"])
        if not args.dry_run:
            result["write"] = materialize_plan(
                plan,
                prepared_root=_resolve(args.prepared_root),
                manifest_path=_resolve(args.manifest_output),
            )
            result["ok"] = True
    except (OSError, TypeError, ValueError) as exc:
        result = {"ok": False, "dry_run": bool(args.dry_run), "error": str(exc)}
    summary_path = _resolve(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if result['ok'] else 'FAIL'}: dry_run={result['dry_run']} "
        f"reviewed={result.get('reviewed_row_count', 0)} missing={result.get('missing_source_count', 0)} "
        f"summary={summary_path}"
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
