#!/usr/bin/env python3
"""Build the gated Notebook 2/6 recovery experiment campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs  # noqa: E402
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402
from src.data.ood_evidence_manifest import read_manifest, validate_manifest_rows  # noqa: E402
from src.ood.recovery_campaign import build_recovery_campaign, render_campaign_markdown  # noqa: E402


def _read(path: Path) -> dict:
    if path.is_file():
        source = path.read_text(encoding="utf-8")
    else:
        relative = path.as_posix()
        source = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_oe_recovery_baseline.json",
    )
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_stage_a.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "architecture" / "adapter_ood_oe_recovery_campaign.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs") / "architecture" / "adapter_ood_oe_recovery_campaign.md",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=DEFAULT_MATERIALIZED_DATASET_ROOT / "adapter_ood_oe_evidence_manifest.csv",
    )
    return parser.parse_args()


def _evidence_state(path: Path) -> tuple[str, set[str]]:
    if not path.is_file():
        return "", set()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        report = validate_manifest_rows(read_manifest(path), verify_files=False)
    except (OSError, ValueError):
        return digest, set()
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    if any(not issue.get("target") for issue in errors):
        return digest, set()
    ready = {
        target
        for target in report["target_role_counts"]
        if report["target_role_counts"][target]
        and not any(issue.get("target") == target for issue in errors)
    }
    return digest, ready


def main() -> int:
    args = parse_args()
    recommendations = get_adapter_recs()
    defaults = {target: dict(item.get("defaults") or {}) for target, item in recommendations.items()}
    evidence_manifest = args.evidence_manifest if args.evidence_manifest.is_absolute() else REPO_ROOT / args.evidence_manifest
    evidence_digest, evidence_ready_targets = _evidence_state(evidence_manifest)
    report = build_recovery_campaign(
        _read(args.baseline),
        _read(args.stage_a),
        seed=args.seed,
        target_defaults=defaults,
        evidence_manifest_digest=evidence_digest,
        evidence_ready_targets=evidence_ready_targets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_campaign_markdown(report), encoding="utf-8")
    print(
        f"PASS: targets={report['target_count']} experiments={report['experiment_count']} "
        f"evidence_ready={report['evidence_ready_target_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
