"""CLI wrapper extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb.

This script runs baseline evaluation and calibration over a router evaluation
dataset and writes JSON outputs. It keeps behavior parity with the notebook
cells but is testable and runnable from CI or the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.utils.reporting import write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calibrate router handoff thresholds")
    p.add_argument("--router-eval-root", default="data/router_eval")
    p.add_argument("--baseline-output", default=".runtime_tmp/router_eval.json")
    p.add_argument("--calibration-output", default=".runtime_tmp/router_calibration.json")
    p.add_argument("--preset", default="quick")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--run-calibration", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Defer imports so repo can be on sys.path like in notebooks
    try:
        from scripts.evaluate_router_surface import discover_eval_samples, evaluate_router_surface
        from scripts.calibrate_router_surface import calibrate_router_surface
    except Exception:
        # If running from a different cwd, insert repo root heuristics similar to the notebook
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scripts.evaluate_router_surface import discover_eval_samples, evaluate_router_surface
        from scripts.calibrate_router_surface import calibrate_router_surface

    root = Path(args.router_eval_root)
    print(f"[CONFIG] eval_root={root}")
    print(f"[CONFIG] device={args.device} preset={args.preset}")

    baseline_out = Path(args.baseline_output)
    calibration_out = Path(args.calibration_output)

    if args.run_baseline:
        print("[RUN] baseline evaluation")
        samples = discover_eval_samples(root)
        baseline = evaluate_router_surface(samples, config_env="colab", device=args.device)
        write_json(baseline_out, baseline)

    if args.run_calibration:
        print("[RUN] calibration")
        calibration = calibrate_router_surface(
            eval_root=root,
            preset=args.preset,
            max_variants=128,
            target_negative_false_accept_rate=0.05,
            max_crop_accuracy_drop=0.02,
            max_part_precision_drop=0.02,
        )
        write_json(calibration_out, calibration)


if __name__ == "__main__":
    main()
