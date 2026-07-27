import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_runtime_real.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_runtime_real", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_summarize_measurements_reports_percentiles():
    summary = _MODULE._summarize_measurements([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 4.0
    assert summary["mean_ms"] == 2.5
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


def test_benchmark_repeated_tracks_status_counts():
    payload = _MODULE._benchmark_repeated(lambda: {"status": "success"}, repeat=3, warmup=1)

    assert payload["summary"]["count"] == 3
    assert payload["status_counts"] == {"success": 3}
    assert payload["last_status"] == "success"
