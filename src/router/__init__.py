"""Lazy router exports to avoid importing heavy optional dependencies at package import time."""

from __future__ import annotations

from typing import Any

__all__ = ["RouterPipeline", "DiagnosticScoutingAnalyzer"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from src.router.diagnostic_scouting import DiagnosticScoutingAnalyzer
    from src.router.router_pipeline import RouterPipeline

    exports = {
        "RouterPipeline": RouterPipeline,
        "DiagnosticScoutingAnalyzer": DiagnosticScoutingAnalyzer,
    }
    return exports[name]
