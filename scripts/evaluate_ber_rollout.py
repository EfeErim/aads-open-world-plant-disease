#!/usr/bin/env python3
"""Compare BER candidate artifact roots against a BER-off baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_artifact_root", help="Artifact root for the BER-off baseline run.")
    parser.add_argument("candidate_artifact_roots", nargs="+", help="One or more BER candidate artifact roots.")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.002)
    parser.add_argument("--min-ood-auroc-gain", type=float, default=0.01)
    parser.add_argument("--min-fpr-improvement", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.training.services.ber_rollout import evaluate_ber_candidate

    args = _parse_args(argv)
    baseline_root = Path(args.baseline_artifact_root)
    exit_code = 0
    for candidate_root_raw in args.candidate_artifact_roots:
        candidate_root = Path(candidate_root_raw)
        result = evaluate_ber_candidate(
            baseline_artifact_root=baseline_root,
            candidate_artifact_root=candidate_root,
            max_accuracy_drop=args.max_accuracy_drop,
            min_ood_auroc_gain=args.min_ood_auroc_gain,
            min_fpr_improvement=args.min_fpr_improvement,
        )
        print(json.dumps({"candidate_artifact_root": str(candidate_root), **result}, indent=2))
        if not result["passed"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
