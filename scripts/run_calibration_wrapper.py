#!/usr/bin/env python3
"""Wrapper to run calibrate_router_surface directly and write to absolute path."""
from pathlib import Path
import json
import sys

# Ensure repo root is on sys.path so scripts.* imports resolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_router_surface import calibrate_router_surface


def main() -> int:
    out = Path(r"D:/bitirme projesi/.runtime_tmp/router_calibration_direct_wrapper.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = calibrate_router_surface(Path("data/router_eval"), config_env="base", preset="quick")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("WROTE_WRAPPER_OUTPUT", out)
        return 0
    except Exception as e:
        print("WRAPPER_FAILED", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
