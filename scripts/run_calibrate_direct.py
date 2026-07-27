from pathlib import Path
import json, traceback, sys, runpy

OUT = Path(r"D:/bitirme projesi/.runtime_tmp/router_calibration_direct.json")
try:
    ns = runpy.run_path("scripts/calibrate_router_surface.py")
    calibrate_router_surface = ns.get("calibrate_router_surface")
    if calibrate_router_surface is None:
        raise RuntimeError("calibrate_router_surface not found in scripts/calibrate_router_surface.py")
    result = calibrate_router_surface(Path("data/router_eval"), config_env="base", preset="quick")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", OUT)
except Exception:
    print("EXCEPTION", file=sys.stderr)
    traceback.print_exc()
    raise
