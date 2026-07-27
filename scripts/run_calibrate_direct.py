import json
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime_tmp" / "router_calibration_direct.json"
try:
    ns = runpy.run_path(str(ROOT / "scripts" / "calibrate_router_surface.py"))
    calibrate_router_surface = ns.get("calibrate_router_surface")
    if calibrate_router_surface is None:
        raise RuntimeError("calibrate_router_surface not found in scripts/calibrate_router_surface.py")
    result = calibrate_router_surface(ROOT / "data" / "router_eval", config_env="base", preset="quick")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", OUT)
except Exception:
    print("EXCEPTION", file=sys.stderr)
    traceback.print_exc()
    raise
