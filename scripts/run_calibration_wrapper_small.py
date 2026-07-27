#!/usr/bin/env python3
"""Run calibrate_router_surface on a small temporary eval root for debugging."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from scripts.calibrate_router_surface import calibrate_router_surface


def main() -> int:
    out = ROOT / ".runtime_tmp" / "router_calibration_direct_small.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = calibrate_router_surface(
            ROOT / "data" / "temp_router_eval",
            config_env="base",
            preset="quick",
            device="cpu",
        )
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("WROTE_WRAPPER_SMALL_OUTPUT", out)
        return 0
    except Exception as e:
        print("WRAPPER_SMALL_FAILED", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
