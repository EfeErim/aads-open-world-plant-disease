from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict

PRESENTATION_STYLES = """
<style>
  .aads-demo {font-family: Arial, sans-serif; color: #102c40; max-width: 920px;}
  .aads-demo h2 {background: #071b2c; color: white; padding: 7px 11px; border-radius: 8px; font-size: 18px; margin: 0 0 7px;}
  .aads-demo h3 {color: #12364d; margin: 8px 0 5px; font-size: 15px;}
  .aads-demo .how-it-works {background: #f5fbfb; border-left: 4px solid #2b7c75; padding: 6px 10px; margin: 6px 0;}
  .aads-demo .how-it-works strong {color: #12364d;}
  .aads-demo .how-it-works ol {margin: 4px 0 2px; padding-left: 20px; columns: 2; column-gap: 24px;}
  .aads-demo .how-it-works li {font-size: 11px; line-height: 1.25; margin: 0 0 4px; break-inside: avoid;}
  .aads-demo .result-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; margin: 5px 0 7px;}
  .aads-demo .result-card {padding: 7px 9px; border-radius: 8px; background: #edf5f7; border-left: 4px solid #2b7c75; font-size: 11px;}
  .aads-demo .result-card strong {display: block; color: #12364d; margin-bottom: 3px;}
  .aads-demo table {border-collapse: collapse; width: 100%; margin: 8px 0 16px;}
  .aads-demo th, .aads-demo td {border: 1px solid #c6d7df; padding: 7px 9px; text-align: left;}
  .aads-demo th {background: #edf5f7; width: 32%;}
  .aads-demo .note {background: #fff4df; padding: 6px 9px; border-left: 4px solid #f4a261; margin: 5px 0; font-size: 11px;}
</style>
"""


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text(value: Any, default: str = "-") -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _audience_ood_result(summary: Dict[str, Any]) -> str:
    if not summary["ood_available"]:
        return "OOD check unavailable."
    if summary["is_ood"]:
        return "Unknown or unsupported image: prediction withheld."
    return "Known-type image: accepted for prediction."


MAX_PRESENTATION_BOXES = 3
PRESENTATION_ROUTER_FIGSIZE = (8.6, 3.3)


def _select_presentation_detections(detections: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return detections[:MAX_PRESENTATION_BOXES]


def build_presentation_summary(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    auto_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract recording-friendly facts from the maintained inference payloads."""
    router_details = _as_dict(router_result.get("router_details"))
    diagnostics = _as_dict(router_result.get("diagnostics"))
    adapter_target = _as_dict(router_result.get("adapter_target"))
    handoff = _as_dict(auto_result.get("router_handoff"))
    ood = _as_dict(auto_result.get("ood_analysis"))
    input_guard = _as_dict(auto_result.get("input_guard"))
    detections = [_as_dict(item) for item in _as_list(router_details.get("detections"))]
    timings = _as_dict(router_details.get("stage_timings_ms"))
    roi_stats = _as_dict(router_details.get("roi_stats"))
    candidates = [_as_dict(item) for item in _as_list(diagnostics.get("top_crop_candidates"))]

    notebook_gate = _as_dict(router_result.get("notebook_gate"))
    if notebook_gate:
        gate_status = "Accepted" if notebook_gate.get("accepted") else "Rejected"
        gate_reason = "; ".join(str(item) for item in _as_list(notebook_gate.get("reasons"))) or "Threshold checks passed."
    else:
        gate_status = "Not applied"
        gate_reason = "Notebook-level crop confidence gate was not applied."

    adapter_ran = bool(handoff.get("adapter_ran"))
    diagnosis = auto_result.get("diagnosis")
    final_status = _text(auto_result.get("status"), "unknown")
    final_message = _text(auto_result.get("message"), "")
    if diagnosis:
        final_decision = f"Model prediction: {diagnosis} ({_float(auto_result.get('confidence')):.3f})"
    elif final_message:
        final_decision = f"No disease prediction: {final_message}"
    else:
        final_decision = f"No disease prediction: status={final_status}"

    return {
        "image_path": str(image_path),
        "pipeline_type": _text(router_details.get("pipeline_type"), "sam3_bioclip25"),
        "processing_time_ms": _float(router_details.get("processing_time_ms")),
        "sam3_instances_raw": int(router_details.get("sam3_instances_raw", router_details.get("sam3_instances", 0)) or 0),
        "sam3_instances_retained": int(router_details.get("sam3_instances_retained", len(detections)) or 0),
        "roi_seen": int(roi_stats.get("seen", 0) or 0),
        "roi_retained": int(roi_stats.get("retained", 0) or 0),
        "roi_classification_calls": int(roi_stats.get("classification_calls", 0) or 0),
        "stage_timings_ms": timings,
        "detections": detections,
        "top_crop_candidates": candidates,
        "crop": _text(router_result.get("crop"), "unknown"),
        "part": _text(router_result.get("part"), "unknown"),
        "router_confidence": _float(router_result.get("router_confidence")),
        "crop_margin": _float(diagnostics.get("crop_confidence_margin")),
        "raw_part_label": _text(diagnostics.get("raw_part_label"), "unknown"),
        "raw_part_confidence": _float(diagnostics.get("raw_part_confidence")),
        "part_unknown_confidence": _float(diagnostics.get("part_unknown_confidence")),
        "part_rejection_reason": _text(diagnostics.get("part_rejection_reason"), ""),
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "adapter_dir": _text(adapter_target.get("adapter_dir")),
        "adapter_exists": bool(adapter_target.get("exists")),
        "adapter_ran": adapter_ran,
        "handoff_status": _text(handoff.get("source_status"), "unknown"),
        "final_status": final_status,
        "final_decision": final_decision,
        "ood_available": bool(ood),
        "ood_method": _text(ood.get("score_method")),
        "ood_score": _float(ood.get("primary_score")),
        "ood_threshold": _float(ood.get("decision_threshold")),
        "is_ood": bool(ood.get("is_ood")),
        "input_guard_available": bool(input_guard),
        "input_guard_decision": _text(input_guard.get("decision")),
        "input_guard_plant_like": bool(input_guard.get("is_plant_like")),
    }


def build_presentation_flow_html(summary: Dict[str, Any]) -> str:
    """Build the top-level visual explanation for a non-technical audience."""
    items = (
        f"<li><strong>SAM3</strong> finds plant regions to inspect: "
        f"{summary['sam3_instances_retained']} retained from {summary['sam3_instances_raw']} proposals.</li>",
        f"<li><strong>BioCLIP-2.5</strong> identifies the crop and plant part: "
        f"{escape(summary['crop'])} / {escape(summary['part'])} ({summary['router_confidence']:.3f}).</li>",
        f"<li><strong>Safety gate</strong> decides whether a specialist adapter can be used: "
        f"{escape(summary['gate_status'])}.</li>",
        f"<li><strong>SD-LoRA adapter + OOD check</strong> returns the model prediction and rejects unsupported images: "
        f"{escape(_audience_ood_result(summary))}</li>",
    )
    return "<div class='how-it-works'><strong>How the system reads this image</strong><ol>" + "".join(items) + "</ol></div>"


def _render_router_figure(summary: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    image = Image.open(summary["image_path"]).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    visible_detections = _select_presentation_detections(summary["detections"])
    for detection in visible_detections:
        bbox = _as_list(detection.get("bbox"))
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [_float(value) for value in bbox[:4]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, x2 = max(0.0, x1), min(width - 1.0, x2)
        y1, y2 = max(0.0, y1), min(height - 1.0, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color = (66, 133, 244)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)

    fig, axes = plt.subplots(1, 2, figsize=PRESENTATION_ROUTER_FIGSIZE)
    axes[0].imshow(image)
    axes[0].set_title("1. Uploaded input image")
    axes[1].imshow(annotated)
    axes[1].set_title(f"2. SAM3 candidate regions for inspection (showing {len(visible_detections)})")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout(pad=0.45)
    plt.show()


def render_presentation_demo(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    auto_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Render one detailed, recording-friendly notebook output from real payloads."""
    from IPython.display import HTML, display

    summary = build_presentation_summary(image_path, router_result=router_result, auto_result=auto_result)
    header = (
        f"{PRESENTATION_STYLES}<div class='aads-demo'><h2>AADS v6 - Detailed Router-to-Adapter Demo</h2>"
        + build_presentation_flow_html(summary)
        + "<p class='note'><strong>Important:</strong> SAM3 boxes are not disease predictions. "
        "They mark candidate plant regions for BioCLIP-2.5 inspection.</p>"
        + "</div>"
    )
    display(HTML(header))

    remaining = (
        "<div class='aads-demo'><h3>Audience Summary</h3>"
        "<div class='result-grid'>"
        f"<div class='result-card'><strong>BioCLIP-2.5 route</strong>Crop: {escape(summary['crop'])}<br>"
        f"Plant part: {escape(summary['part'])}</div>"
        f"<div class='result-card'><strong>Safety decision</strong>{escape(summary['gate_status'])}<br>"
        f"Specialist adapter loaded: {_yes_no(summary['adapter_ran'])}</div>"
        f"<div class='result-card'><strong>Adapter model prediction (not a verified diagnosis)</strong>"
        f"{escape(summary['final_decision'])}</div>"
        f"<div class='result-card'><strong>OOD check</strong>{escape(_audience_ood_result(summary))}</div>"
        "</div>"
        + "</div>"
    )
    display(HTML(remaining))
    _render_router_figure(summary)
    return summary


__all__ = [
    "PRESENTATION_STYLES",
    "PRESENTATION_ROUTER_FIGSIZE",
    "build_presentation_flow_html",
    "build_presentation_summary",
    "render_presentation_demo",
]
