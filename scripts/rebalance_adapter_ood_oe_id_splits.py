from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapter_ood_oe_split_rebalancing import rebalance_runtime_id_splits  # noqa: E402


def _supplemental_counts(manifest_path: Path, target: str) -> dict[str, dict[str, int]]:
    role_to_split = {"id_train": "continual", "id_val": "val", "id_test": "test"}
    families: dict[tuple[str, str], set[str]] = defaultdict(set)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target") != target or row.get("source") == "prepared_runtime_dataset":
                continue
            split = role_to_split.get(str(row.get("role") or ""))
            if split:
                families[(str(row.get("disease_id") or ""), split)].add(
                    str(row.get("evidence_family_id") or row.get("relative_path") or "")
                )
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for (disease_id, split), family_ids in families.items():
        counts[disease_id][split] = len(family_ids)
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebalance runtime ID splits without splitting evidence families.")
    parser.add_argument("dataset_roots", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="Move files and update split manifests.")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("data/prepared_runtime_datasets/adapter_ood_oe_evidence_manifest.csv"),
    )
    args = parser.parse_args()

    reports = [
        rebalance_runtime_id_splits(
            root,
            dry_run=not args.write,
            supplemental_counts=_supplemental_counts(args.evidence_manifest, root.name),
        )
        for root in args.dataset_roots
    ]
    payload = {"ok": all(report["ok"] for report in reports), "reports": reports}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
