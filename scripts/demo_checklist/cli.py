"""CLI parser for the M2 demo checklist runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", type=Path, default=Path("docs/demo_checklist.md"))
    parser.add_argument("--no-checklist", action="store_true", help="Run only rows from --extra-manifest files.")
    parser.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        type=Path,
        help="CSV manifest with image_id, source, expected_target, expected_behavior, notes columns.",
    )
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/m2_demo_checklist_run.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path(".runtime_tmp/m2_demo_checklist_run.md"))
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--analysis-markdown-output", type=Path)
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-prototype-reconciler", action="store_true")
    parser.add_argument("--prototype-bank", type=Path)
    parser.add_argument("--taxonomy-registry", type=Path)
    parser.add_argument("--prototype-calibration-report", type=Path)
    parser.add_argument("--prototype-min-similarity", type=float)
    parser.add_argument("--prototype-min-margin", type=float)
    parser.add_argument("--prototype-min-negative-gap", type=float)
    parser.add_argument(
        "--handoff-cache",
        type=Path,
        default=Path(".runtime_tmp/m2_router_prototype_handoff_cache.json"),
        help="JSON cache for per-image router/prototype handoff outputs in official batched mode.",
    )
    parser.add_argument(
        "--refresh-handoff-cache",
        action="store_true",
        help="Ignore existing router/prototype handoff cache entries and rewrite the cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Official mode router batch size.",
    )
    parser.add_argument(
        "--adapter-batch-size",
        type=int,
        default=1,
        help="Official mode adapter batch size after router handoff. Falls back to per-row if unsafe.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-local", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("official", "adapter-smoke", "asset-audit"),
        default="official",
        help=(
            "official runs the Notebook 8 helper path; adapter-smoke skips router with expected crop/part "
            "and is not an official M2 pass; asset-audit only checks files."
        ),
    )
    parser.add_argument(
        "--trust-expected-target",
        action="store_true",
        help="Deprecated alias for --mode adapter-smoke.",
    )
    parser.add_argument("--stop-on-dependency-blocker", action="store_true")
    return parser
