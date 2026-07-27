from typing import Any, Dict, Optional


def emit_event(telemetry: Any, event_type: str, payload: Dict[str, Any], *, phase: str, force_sync: Optional[bool] = None) -> None:
    """Call telemetry.emit_event if available, with an optional force_sync flag."""
    if telemetry is None:
        return
    emit = getattr(telemetry, "emit_event", None)
    if not callable(emit):
        return
    if force_sync is None:
        emit(event_type, payload, phase=phase)
        return
    emit(event_type, payload, phase=phase, force_sync=force_sync)


def update_latest(telemetry: Any, payload: Dict[str, Any]) -> None:
    """Call telemetry.update_latest if available."""
    if telemetry is None:
        return
    updater = getattr(telemetry, "update_latest", None)
    if callable(updater):
        try:
            updater(payload)
        except Exception:
            # Telemetry should not break main flow
            return
