#!/usr/bin/env python3
"""Validate the strict adapter ID/OE/OOD evidence manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ood_evidence_manifest import MANIFEST_SCHEMA, read_manifest, validate_manifest_rows  # noqa: E402
from src.ood.recovery import TARGET_ADAPTERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument(
        "--file-path-prefix",
        default="",
        help="Strip this required logical prefix before resolving manifest members under --base-dir.",
    )
    parser.add_argument("--no-verify-files", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime_tmp") / "adapter_ood_oe_manifest_validation.json",
    )
    parser.add_argument("--min-id-test", type=int, default=30)
    parser.add_argument("--min-ood-test", type=int, default=30)
    parser.add_argument("--min-per-ood-type", type=int, default=5)
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGET_ADAPTERS,
        help="Validate only the selected target; repeat for a bounded multi-target smoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_manifest(args.manifest)
        base_dir = args.base_dir or args.manifest.parent
        report = validate_manifest_rows(
            rows,
            required_targets=args.target,
            base_dir=base_dir,
            file_path_prefix=args.file_path_prefix,
            verify_files=not args.no_verify_files,
            min_id_test=args.min_id_test,
            min_ood_test=args.min_ood_test,
            min_per_ood_type=args.min_per_ood_type,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": MANIFEST_SCHEMA,
            "ok": False,
            "error_count": 1,
            "issues": [{"severity": "error", "code": "manifest_unreadable", "message": str(exc)}],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if report['ok'] else 'FAIL'}: rows={report.get('row_count', 0)} "
        f"errors={report.get('error_count', 0)} report={args.output}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
