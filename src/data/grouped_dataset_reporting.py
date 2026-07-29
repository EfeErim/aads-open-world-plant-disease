"""Human-review packets and summaries for grouped dataset preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.data.grouped_dataset_policy import (
    BIOCLIP_AUTO_MIN,
    BIOCLIP_CROSS_CLASS_BLOCK_MIN,
    BIOCLIP_REVIEW_MIN,
    DINO_AUTO_MIN,
    DINO_CROSS_CLASS_BLOCK_MIN,
    DINO_REVIEW_MIN,
    HUMAN_REVIEW_PACKET_FILENAME,
    LABEL_REVIEW_SUMMARY_FILENAME,
    PHASH_AUTO_MAX_DISTANCE,
    PHASH_REVIEW_MAX_DISTANCE,
)
from src.shared.csv_utils import read_csv_preview as _read_csv_preview


def _coerce_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _skipped_class_entry(class_name: str, reason: str, health_entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_name": class_name,
        "reason": reason,
        "image_count": int(health_entry["image_count"]),
        "family_count": int(health_entry["family_count"]),
        "eval_eligible_family_count": int(health_entry["eval_eligible_family_count"]),
    }


def build_human_review_packet(
    summary: Dict[str, Any],
    *,
    artifact_root: Path,
    max_review_items: int = 25,
) -> Dict[str, Any]:
    """Build the compact human-in-loop decision packet for Notebook 0.

    The packet is intentionally conservative: it highlights audit conditions that
    can make a benchmark misleading, but it does not relabel images or override
    the split plan. Notebook 0 uses it to pause only around high-impact decisions.
    """

    artifact_root = Path(artifact_root)
    counts = dict(summary.get("summary", {}) or {})
    blocking_issues = [str(item) for item in list(summary.get("blocking_issues") or []) if str(item).strip()]
    skipped_classes = list(summary.get("skipped_classes") or [])
    try:
        max_rows = max(0, int(max_review_items))
    except (TypeError, ValueError):
        max_rows = 25

    cross_class_count, cross_class_preview = _read_csv_preview(
        artifact_root / "cross_class_conflicts.csv",
        max_rows=max_rows,
        fields=("class_a", "class_b", "path_a", "path_b", "reason", "phash_distance"),
    )
    high_risk_count, high_risk_preview = _read_csv_preview(
        artifact_root / "same_class_high_risk_clusters.csv",
        max_rows=max_rows,
        fields=("cluster_id", "normalized_class_name", "image_count", "reason", "relative_paths"),
    )
    label_review_count, label_review_preview = _read_csv_preview(
        artifact_root / "label_review_candidates.csv",
        max_rows=max_rows,
        fields=("normalized_class_name", "relative_path", "label_risk_level", "label_risk_score", "label_risk_reason"),
    )
    source_style_count, source_style_preview = _read_csv_preview(
        artifact_root / "source_style_groups.csv",
        max_rows=max_rows,
        fields=("source_style_group", "source_style_risk", "source_style_reason", "image_count", "relative_paths"),
    )

    cross_class_count = max(cross_class_count, _coerce_count(counts.get("cross_class_conflicts")))
    high_risk_count = max(high_risk_count, _coerce_count(counts.get("same_class_high_risk_clusters")))
    label_review_count = max(label_review_count, _coerce_count(counts.get("label_review_candidates")))

    decision_points: List[Dict[str, Any]] = []
    if blocking_issues or cross_class_count:
        decision_points.append(
            {
                "id": "blocking_conflicts_or_split_blockers",
                "severity": "critical",
                "title": "Direct materialization is not safe yet.",
                "reason": (
                    "Cross-class conflicts or split blockers can contaminate the benchmark. "
                    "Use the prepared working copy cleanup path, fix the source dataset, or stop."
                ),
                "default_decision": "stop_direct_materialization",
                "counts": {
                    "blocking_issues": len(blocking_issues),
                    "cross_class_conflicts": cross_class_count,
                },
                "artifacts": ["cross_class_conflicts.csv", "class_health_report.json"],
                "preview": {
                    "blocking_issues": blocking_issues[:max_rows],
                    "cross_class_conflicts": cross_class_preview,
                },
            }
        )

    if label_review_count or high_risk_count:
        decision_points.append(
            {
                "id": "label_or_family_review_queue",
                "severity": "high",
                "title": "Review candidates were found.",
                "reason": (
                    "DINOv3/BioCLIP similarity and hash-family checks found ambiguous samples. "
                    "The safe default is to keep uncertain non-blocking items out of canonical val/test."
                ),
                "default_decision": "continue_with_train_only_routing",
                "counts": {
                    "label_review_candidates": label_review_count,
                    "same_class_high_risk_clusters": high_risk_count,
                },
                "artifacts": ["label_review_candidates.csv", "same_class_high_risk_clusters.csv"],
                "preview": {
                    "label_review_candidates": label_review_preview,
                    "same_class_high_risk_clusters": high_risk_preview,
                },
            }
        )

    source_style_risk_images = _coerce_count(counts.get("source_style_risk_images"))
    train_only_routed_images = _coerce_count(counts.get("train_only_routed_images"))
    if source_style_risk_images or train_only_routed_images:
        decision_points.append(
            {
                "id": "source_style_or_train_only_routing",
                "severity": "medium",
                "title": "Some samples were routed away from canonical evaluation.",
                "reason": (
                    "Source-style, synthetic, eval-quality, or label-risk cues were treated as benchmark-risk signals. "
                    "The samples remain usable for continual training unless they are blocking conflicts."
                ),
                "default_decision": "continue_with_conservative_eval_filter",
                "counts": {
                    "source_style_risk_images": source_style_risk_images,
                    "train_only_routed_images": train_only_routed_images,
                    "source_style_groups": source_style_count,
                },
                "artifacts": ["source_style_groups.csv", "family_manifest.csv"],
                "preview": {
                    "source_style_groups": source_style_preview,
                },
            }
        )

    if skipped_classes:
        decision_points.append(
            {
                "id": "class_scope_changed",
                "severity": "medium",
                "title": "One or more classes were skipped.",
                "reason": "Skipped classes did not retain enough clean evaluation families for the runtime split contract.",
                "default_decision": "continue_only_if_scope_is_expected",
                "counts": {"skipped_classes": len(skipped_classes)},
                "artifacts": ["class_health_report.json"],
                "preview": {"skipped_classes": skipped_classes[:max_rows]},
            }
        )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    decision_points = sorted(
        decision_points,
        key=lambda item: (severity_order.get(str(item.get("severity", "low")), 99), str(item.get("id", ""))),
    )
    pause_recommended = bool(decision_points)
    if any(point.get("severity") == "critical" for point in decision_points):
        recommended_action = "prepare_clean_working_copy_or_stop"
        safe_default_decision = "do_not_materialize_directly"
    elif any(point.get("severity") == "high" for point in decision_points):
        recommended_action = "confirm_train_only_routing_before_materialization"
        safe_default_decision = "continue_with_conservative_train_only_routing"
    elif decision_points:
        recommended_action = "confirm_benchmark_scope_before_materialization"
        safe_default_decision = "continue_with_conservative_eval_filter"
    else:
        recommended_action = "continue"
        safe_default_decision = "continue"

    return {
        "schema_version": "v1_human_review_packet",
        "runtime_ready": bool(summary.get("runtime_ready")),
        "pause_recommended": pause_recommended,
        "recommended_action": recommended_action,
        "safe_default_decision": safe_default_decision,
        "max_review_items": max_rows,
        "artifact_root": str(artifact_root),
        "threshold_policy": {
            "calibration_mode": "fixed_conservative_defaults",
            "note": (
                "These thresholds are repo heuristics used to create review and routing evidence; "
                "the packet asks for human confirmation instead of claiming ground-truth relabeling."
            ),
            "phash_auto_max_distance": PHASH_AUTO_MAX_DISTANCE,
            "phash_review_max_distance": PHASH_REVIEW_MAX_DISTANCE,
            "dino_auto_min": DINO_AUTO_MIN,
            "dino_review_min": DINO_REVIEW_MIN,
            "bioclip_auto_min": BIOCLIP_AUTO_MIN,
            "bioclip_review_min": BIOCLIP_REVIEW_MIN,
            "dino_cross_class_block_min": DINO_CROSS_CLASS_BLOCK_MIN,
            "bioclip_cross_class_block_min": BIOCLIP_CROSS_CLASS_BLOCK_MIN,
        },
        "counts": {
            "blocking_issues": len(blocking_issues),
            "cross_class_conflicts": cross_class_count,
            "same_class_high_risk_clusters": high_risk_count,
            "label_review_candidates": label_review_count,
            "source_style_risk_images": source_style_risk_images,
            "train_only_routed_images": train_only_routed_images,
            "skipped_classes": len(skipped_classes),
        },
        "decision_points": decision_points,
        "review_artifacts": [
            "human_review_packet.json",
            LABEL_REVIEW_SUMMARY_FILENAME,
            "prep_summary.json",
            "class_health_report.json",
            "label_review_candidates.csv",
            "same_class_high_risk_clusters.csv",
            "cross_class_conflicts.csv",
            "source_style_groups.csv",
        ],
    }


def build_label_review_summary(
    summary: Dict[str, Any],
    *,
    label_risk_summary: Dict[str, Any],
    human_review_packet: Dict[str, Any],
    label_review_candidates: Sequence[Dict[str, Any]],
    max_preview_items: int = 10,
) -> Dict[str, Any]:
    """Build the Notebook 0 label-quality summary anchored on the human review gate."""

    nested_summary = dict(summary.get("summary", {}) or {})
    review_preview = [dict(item) for item in list(label_review_candidates)[: max(0, int(max_preview_items))]]
    return {
        "schema_version": "v1_label_review_summary",
        "surface": "notebook_0_prepare_grouped_dataset_for_training",
        "runtime_ready": bool(summary.get("runtime_ready")),
        "crop_name": str(summary.get("crop_name", "") or ""),
        "part_name": str(summary.get("part_name", "") or ""),
        "source_root": str(summary.get("source_root", "") or ""),
        "prepared_runtime_root": str(summary.get("prepared_runtime_root", "") or ""),
        "human_in_the_loop": {
            "enabled": True,
            "pause_recommended": bool(human_review_packet.get("pause_recommended")),
            "recommended_action": str(human_review_packet.get("recommended_action", "") or ""),
            "safe_default_decision": str(human_review_packet.get("safe_default_decision", "") or ""),
            "review_artifacts": list(human_review_packet.get("review_artifacts", []) or []),
        },
        "counts": {
            "label_review_candidates": int(nested_summary.get("label_review_candidates", 0) or 0),
            "label_train_only_risk_images": int(nested_summary.get("label_train_only_risk_images", 0) or 0),
            "label_blocking_conflict_images": int(nested_summary.get("label_blocking_conflict_images", 0) or 0),
            "same_class_high_risk_clusters": int(nested_summary.get("same_class_high_risk_clusters", 0) or 0),
            "cross_class_conflicts": int(nested_summary.get("cross_class_conflicts", 0) or 0),
            "train_only_routed_images": int(nested_summary.get("train_only_routed_images", 0) or 0),
            "source_style_risk_images": int(nested_summary.get("source_style_risk_images", 0) or 0),
            "skipped_classes": int(nested_summary.get("skipped_classes", 0) or 0),
        },
        "label_risk_levels": dict(label_risk_summary.get("level_counts", {}) or {}),
        "policy": dict(label_risk_summary.get("policy", {}) or {}),
        "signals": dict(label_risk_summary.get("signals", {}) or {}),
        "review_queue": {
            "path": "label_review_candidates.csv",
            "candidate_count": int(label_risk_summary.get("review_candidate_count", 0) or 0),
            "preview": review_preview,
        },
        "artifacts": {
            "human_review_packet_json": HUMAN_REVIEW_PACKET_FILENAME,
            "label_risk_summary_json": "label_risk_summary.json",
            "label_review_candidates_csv": "label_review_candidates.csv",
            "same_class_high_risk_clusters_csv": "same_class_high_risk_clusters.csv",
            "cross_class_conflicts_csv": "cross_class_conflicts.csv",
            "class_health_report_json": "class_health_report.json",
        },
        "note": (
            "This is the Notebook 0 audit-time label-quality surface. It summarizes heuristic label-risk routing "
            "and the human review gate before runtime-dataset materialization. It does not auto-relabel samples."
        ),
    }


def format_human_review_packet(packet: Dict[str, Any]) -> str:
    """Render a compact console summary for Notebook 0 review prompts."""

    counts = dict(packet.get("counts", {}) or {})
    lines = [
        "[HUMAN REVIEW] Notebook 0 audit gate",
        f"  runtime_ready={packet.get('runtime_ready')} pause_recommended={packet.get('pause_recommended')}",
        f"  recommended_action={packet.get('recommended_action')} safe_default={packet.get('safe_default_decision')}",
        (
            "  counts="
            f"blocking_issues={counts.get('blocking_issues', 0)} "
            f"cross_class_conflicts={counts.get('cross_class_conflicts', 0)} "
            f"label_review_candidates={counts.get('label_review_candidates', 0)} "
            f"high_risk_clusters={counts.get('same_class_high_risk_clusters', 0)} "
            f"source_style_risk_images={counts.get('source_style_risk_images', 0)} "
            f"train_only_routed_images={counts.get('train_only_routed_images', 0)} "
            f"skipped_classes={counts.get('skipped_classes', 0)}"
        ),
        "  artifacts=" + ", ".join(str(item) for item in packet.get("review_artifacts", [])[:7]),
    ]
    decision_points = list(packet.get("decision_points") or [])
    if not decision_points:
        lines.append("  decision_points=none")
        return "\n".join(lines)

    lines.append("  decision_points:")
    for point in decision_points:
        point_counts = dict(point.get("counts", {}) or {})
        count_text = ", ".join(f"{key}={value}" for key, value in point_counts.items())
        lines.append(
            f"   - {point.get('severity')}:{point.get('id')} "
            f"default={point.get('default_decision')} counts=({count_text})"
        )
    return "\n".join(lines)
