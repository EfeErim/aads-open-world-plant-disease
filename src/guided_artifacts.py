"""
Human-readable guided artifact indexes for notebook and workflow outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

from src.guided_artifact_specs import (
    PREP_CATEGORY_DOCS,
    PREP_ENTRY_SPECS,
    PREP_START_HERE_LINES,
    TRAINING_CATEGORY_DOCS,
    TRAINING_CURVE_GLOB_SPEC,
    TRAINING_FOLD_ENTRY_SPECS,
    TRAINING_SPLIT_ENTRY_SPECS,
    TRAINING_SPLIT_ORDER_BASES,
    TRAINING_START_HERE_LINES,
    TRAINING_STATIC_ENTRY_SPECS,
)
from src.shared.json_utils import read_json, write_json

_PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

_GUIDED_START_HERE = "guided/00_start_here.md"
_GUIDED_CATALOG = "guided/02_file_catalog.json"
_MOJIBAKE_MARKERS = ("\u00c3", "\u00c5", "\u00c4")


def _repair_mojibake_text(value: str) -> str:
    text = str(value)
    if any(marker in text for marker in _MOJIBAKE_MARKERS):
        current_marker_count = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
        for source_encoding in ("cp1252", "latin-1"):
            try:
                candidate = text.encode(source_encoding).decode("utf-8")
            except UnicodeError:
                continue
            candidate_marker_count = sum(candidate.count(marker) for marker in _MOJIBAKE_MARKERS)
            if candidate_marker_count < current_marker_count:
                text = candidate
                current_marker_count = candidate_marker_count
    return text


def _repair_mojibake_tree(value: Any) -> Any:
    if isinstance(value, str):
        return _repair_mojibake_text(value)
    if isinstance(value, list):
        return [_repair_mojibake_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_mojibake_tree(item) for key, item in value.items()}
    return value


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        try:
            return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
        except (OSError, ValueError):
            return path.as_posix()


def _path_format(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "file"


def _entry_from_path(
    path: Path,
    *,
    base_dir: Path,
    category: str,
    priority: str,
    title_tr: str,
    description_tr: str,
    reader_goal: str,
    generated_by: str,
    decision_importance: str,
    read_order: int,
) -> Dict[str, Any]:
    return {
        "relative_path": _relative_path(path, base_dir),
        "category": str(category),
        "priority": str(priority),
        "title_tr": str(title_tr),
        "description_tr": str(description_tr),
        "reader_goal": str(reader_goal),
        "format": _path_format(path),
        "generated_by": str(generated_by),
        "decision_importance": str(decision_importance),
        "read_order": int(read_order),
        "exists": bool(path.exists()),
        "is_directory": bool(path.is_dir()),
        "size_bytes": None if not path.exists() or path.is_dir() else int(path.stat().st_size),
    }


def _coerce_extra_entry(entry: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    raw_path = entry.get("path")
    if not raw_path:
        raise ValueError("Extra guided entry requires a 'path' field.")
    resolved_path = Path(raw_path)
    return _entry_from_path(
        resolved_path,
        base_dir=base_dir,
        category=str(entry.get("category", "supporting")),
        priority=str(entry.get("priority", "medium")),
        title_tr=str(entry.get("title_tr", resolved_path.name)),
        description_tr=str(entry.get("description_tr", "")),
        reader_goal=str(entry.get("reader_goal", "")),
        generated_by=str(entry.get("generated_by", "guided_artifact_catalog")),
        decision_importance=str(entry.get("decision_importance", "supporting_context")),
        read_order=int(entry.get("read_order", 999)),
    )


def _dedupe_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        relative_path = str(entry.get("relative_path", "")).strip().replace("\\", "/")
        if not relative_path:
            continue
        normalized = dict(entry)
        normalized["relative_path"] = relative_path
        merged[relative_path] = {**merged.get(relative_path, {}), **normalized}
    return sorted(
        merged.values(),
        key=lambda item: (
            _PRIORITY_ORDER.get(str(item.get("priority", "low")), 99),
            int(item.get("read_order", 999)),
            str(item.get("relative_path", "")),
        ),
    )


def _load_training_overview(artifact_root: Path) -> Dict[str, Any]:
    summary = read_json(artifact_root / "training" / "summary.json", default={}, expect_type=dict)
    acceptance = read_json(artifact_root / "adapter_behavioral_acceptance.json", default={}, expect_type=dict)
    readiness = read_json(artifact_root / "production_readiness.json", default={}, expect_type=dict)
    benchmark = read_json(artifact_root / "ood_benchmark" / "summary.json", default={}, expect_type=dict)
    payload = {
        "run_id": summary.get("run_id", ""),
        "run_label": summary.get("run_label", summary.get("run_id", "")),
        "surface": summary.get("surface", ""),
        "crop_name": summary.get("crop_name", ""),
        "part_name": summary.get("part_name", "unspecified"),
        "dataset_key": summary.get("dataset_key", ""),
        "decision_authority": "adapter_behavioral_acceptance",
        "behavioral_acceptance_status": dict(acceptance).get("status", ""),
        "behavioral_acceptance_passed": dict(acceptance).get("pass"),
        "classification_split": dict(readiness).get("classification_split", ""),
        "readiness_status": dict(readiness).get("status", ""),
        "readiness_passed": dict(readiness).get("passed"),
        "ood_evidence_source": dict(readiness).get("ood_evidence_source", summary.get("ood_evidence_source", "")),
        "class_count": summary.get("class_count"),
        "class_names": list(summary.get("class_names", [])) if isinstance(summary.get("class_names"), list) else [],
        "adapter_dir": summary.get("adapter_dir", ""),
        "artifact_dir": summary.get("artifact_dir", ""),
        "checkpoint_count": summary.get("checkpoint_count"),
        "final_metrics": (
            dict(summary.get("final_metrics", {}))
            if isinstance(summary.get("final_metrics"), dict)
            else {}
        ),
        "ood_benchmark_status": dict(benchmark).get("status", ""),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _load_prep_overview(artifact_root: Path) -> Dict[str, Any]:
    summary = read_json(artifact_root / "prep_summary.json", default={}, expect_type=dict)
    human_review = read_json(artifact_root / "human_review_packet.json", default={}, expect_type=dict)
    nested_summary = dict(summary.get("summary", {})) if isinstance(summary.get("summary"), dict) else {}
    payload = {
        "crop_name": summary.get("crop_name", ""),
        "part_name": summary.get("part_name", ""),
        "source_root": summary.get("source_root", ""),
        "runtime_ready": summary.get("runtime_ready"),
        "human_review_pause_recommended": human_review.get("pause_recommended"),
        "human_review_recommended_action": human_review.get("recommended_action", ""),
        "human_review_safe_default": human_review.get("safe_default_decision", ""),
        "prepared_runtime_root": summary.get("prepared_runtime_root", ""),
        "blocking_issue_count": nested_summary.get("blocking_issues"),
        "readable_images": nested_summary.get("readable_images"),
        "excluded_images": nested_summary.get("excluded_images"),
        "same_class_review_pairs": nested_summary.get("same_class_review_pairs"),
        "cross_class_conflicts": nested_summary.get("cross_class_conflicts"),
        "source_style_risk_images": nested_summary.get("source_style_risk_images"),
        "label_review_candidates": nested_summary.get("label_review_candidates"),
        "train_only_routed_images": nested_summary.get("train_only_routed_images"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _append_spec_entry(
    entries: List[Dict[str, Any]],
    *,
    artifact_root: Path,
    base_dir: Path,
    generated_by: str,
    spec: Dict[str, Any],
    substitutions: Dict[str, Any] | None = None,
) -> None:
    params = dict(substitutions or {})
    relative_path = str(spec.get("relative_path", "")).format(**params)
    path = artifact_root / relative_path
    if not path.exists():
        return
    entries.append(
        _entry_from_path(
            path,
            base_dir=base_dir,
            category=str(spec.get("category", "")).format(**params),
            priority=str(spec.get("priority", "medium")),
            title_tr=str(spec.get("title_tr", path.name)).format(**params),
            description_tr=str(spec.get("description_tr", "")).format(**params),
            reader_goal=str(spec.get("reader_goal", "")).format(**params),
            generated_by=generated_by,
            decision_importance=str(spec.get("decision_importance", "supporting_context")),
            read_order=int(spec.get("read_order", 999)),
        )
    )


def _render_start_here(lines: Sequence[str], *, base_dir: Path) -> str:
    return "\n".join(str(line).format(catalog_base=base_dir.resolve().as_posix()) for line in lines)


def _find_training_entries(artifact_root: Path, *, base_dir: Path, generated_by: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for spec in TRAINING_STATIC_ENTRY_SPECS:
        _append_spec_entry(
            entries,
            artifact_root=artifact_root,
            base_dir=base_dir,
            generated_by=generated_by,
            spec=spec,
        )

    for curve_path in sorted((artifact_root / "training").glob(str(TRAINING_CURVE_GLOB_SPEC["glob"]).split("/", 1)[1])):
        entries.append(
            _entry_from_path(
                curve_path,
                base_dir=base_dir,
                category=str(TRAINING_CURVE_GLOB_SPEC["category"]),
                priority=str(TRAINING_CURVE_GLOB_SPEC["priority"]),
                title_tr=str(TRAINING_CURVE_GLOB_SPEC["title_template"]).format(name=curve_path.name),
                description_tr=str(TRAINING_CURVE_GLOB_SPEC["description_tr"]),
                reader_goal=str(TRAINING_CURVE_GLOB_SPEC["reader_goal"]),
                generated_by=generated_by,
                decision_importance=str(TRAINING_CURVE_GLOB_SPEC["decision_importance"]),
                read_order=int(cast(int, TRAINING_CURVE_GLOB_SPEC.get("read_order", 999) or 999)),
            )
        )

    for split_name, order_base in TRAINING_SPLIT_ORDER_BASES.items():
        substitutions = {
            "split": split_name,
            "split_title": split_name.title(),
        }
        for spec in TRAINING_SPLIT_ENTRY_SPECS:
            spec_with_order = {
                **spec,
                "read_order": int(cast(int, order_base or 0)) + int(cast(int, spec.get("read_order_offset", 0) or 0)),
            }
            _append_spec_entry(
                entries,
                artifact_root=artifact_root,
                base_dir=base_dir,
                generated_by=generated_by,
                spec=spec_with_order,
                substitutions=substitutions,
            )

    for fold_dir in sorted((artifact_root / "ood_benchmark" / "folds").glob("*")):
        if not fold_dir.is_dir():
            continue
        for spec in TRAINING_FOLD_ENTRY_SPECS:
            target = fold_dir / str(spec.get("filename", ""))
            if not target.exists():
                continue
            entries.append(
                _entry_from_path(
                    target,
                    base_dir=base_dir,
                    category="ood_and_readiness",
                    priority=str(spec.get("priority", "low")),
                    title_tr=str(spec.get("title_template", target.name)).format(fold_name=fold_dir.name),
                    description_tr=str(spec.get("description_template", "")).format(fold_name=fold_dir.name),
                    reader_goal=str(spec.get("reader_goal", "")),
                    generated_by=generated_by,
                    decision_importance=str(spec.get("decision_importance", "runtime_diagnostic")),
                    read_order=int(cast(int, spec.get("read_order", 999) or 999)),
                )
            )
    return entries


def _find_prep_entries(artifact_root: Path, *, base_dir: Path, generated_by: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for spec in PREP_ENTRY_SPECS:
        _append_spec_entry(
            entries,
            artifact_root=artifact_root,
            base_dir=base_dir,
            generated_by=generated_by,
            spec=spec,
        )
    return entries


def _render_section_markdown(
    *,
    heading: str,
    intro: str,
    entries: Sequence[Dict[str, Any]],
) -> str:
    lines = [f"# {heading}", "", intro.strip(), ""]
    if not entries:
        lines.extend(["Bu bölüm için mevcut artefact bulunamadı.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for entry in entries:
        lines.append(f"## {entry['title_tr']}")
        lines.append(f"- Yol: `{entry['relative_path']}`")
        lines.append(f"- Öncelik: `{entry['priority']}`")
        lines.append(f"- Format: `{entry['format']}`")
        lines.append(f"- Amaç: {entry['reader_goal']}")
        lines.append(f"- Açıklama: {entry['description_tr']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_guided_file(path: Path, content: str | Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        return write_json(path, _repair_mojibake_tree(content), ensure_ascii=False, sort_keys=False)
    path.write_text(_repair_mojibake_text(str(content)), encoding="utf-8")
    return path


def _copy_guided_to_telemetry(telemetry: Any, *, artifact_root: Path, written_paths: Sequence[Path]) -> None:
    if telemetry is None or not hasattr(telemetry, "copy_artifact_file"):
        return
    for path in written_paths:
        relative_path = path.relative_to(artifact_root).as_posix()
        telemetry.copy_artifact_file(path, relative_path)


def _prepare_guided_catalog(
    *,
    root: Path,
    base_dir: Path,
    overview_loader: Any,
    entry_finder: Any,
    overview_updates: Dict[str, Any] | None,
    extra_entries: Sequence[Dict[str, Any]] | None,
    generated_by: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    overview = {**overview_loader(root), **dict(overview_updates or {})}
    catalog_entries = entry_finder(root, base_dir=base_dir, generated_by=generated_by)
    for item in list(extra_entries or []):
        catalog_entries.append(_coerce_extra_entry(dict(item), base_dir=base_dir))
    return overview, _dedupe_entries(catalog_entries)


def _build_guided_payloads(
    *,
    base_dir: Path,
    catalog_kind: str,
    overview_kind: str,
    overview_schema_version: str,
    overview: Dict[str, Any],
    overview_filename: str,
    primary_files: Dict[str, str],
    catalog_entries: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    catalog_payload = {
        "schema_version": "v1_guided_file_catalog",
        "catalog_kind": catalog_kind,
        "catalog_base": str(base_dir.resolve()),
        "entry_count": len(catalog_entries),
        "entries": list(catalog_entries),
    }
    overview_payload = {
        "schema_version": overview_schema_version,
        "overview_kind": overview_kind,
        "catalog_base": str(base_dir.resolve()),
        "primary_files": {
            "start_here": _GUIDED_START_HERE,
            overview_filename.split("/")[-1].replace(".json", "").replace("01_", ""): overview_filename,
            "file_catalog": _GUIDED_CATALOG,
            **primary_files,
        },
        **overview,
    }
    return _repair_mojibake_tree(catalog_payload), _repair_mojibake_tree(overview_payload)


def _write_guided_bundle(
    *,
    guided_dir: Path,
    overview_filename: str,
    overview_payload: Dict[str, Any],
    catalog_payload: Dict[str, Any],
    start_here: str,
    category_to_doc: Dict[str, tuple[str, str, str]],
    catalog_entries: Sequence[Dict[str, Any]],
) -> List[Path]:
    written_paths = [
        _write_guided_file(guided_dir / "00_start_here.md", _repair_mojibake_text(start_here.rstrip()) + "\n"),
        _write_guided_file(guided_dir / Path(overview_filename).name, overview_payload),
        _write_guided_file(guided_dir / "02_file_catalog.json", catalog_payload),
    ]
    for category, (filename, heading, intro) in category_to_doc.items():
        written_paths.append(
            _write_guided_file(
                guided_dir / filename,
                _render_section_markdown(
                    heading=heading,
                    intro=intro,
                    entries=[entry for entry in catalog_entries if entry.get("category") == category],
                ),
            )
        )
    return written_paths


def _merge_guided_telemetry(
    *,
    telemetry: Any,
    artifact_root: Path,
    written_paths: Sequence[Path],
    catalog_entries: Sequence[Dict[str, Any]],
    overview_payload: Dict[str, Any],
    overview_key: str,
    overview_filename: str,
) -> None:
    _copy_guided_to_telemetry(telemetry, artifact_root=artifact_root, written_paths=written_paths)
    if telemetry is not None and hasattr(telemetry, "merge_artifact_catalog"):
        telemetry.merge_artifact_catalog(catalog_entries)
    if telemetry is not None and hasattr(telemetry, "merge_summary_metadata"):
        telemetry.merge_summary_metadata(
            {
                "guided_artifacts": {
                    "start_here": _GUIDED_START_HERE,
                    overview_key: overview_filename,
                    "file_catalog": _GUIDED_CATALOG,
                },
                "guided_catalog_entry_count": len(catalog_entries),
                overview_key: overview_payload,
            }
        )


def refresh_training_guided_artifacts(
    artifact_root: str | Path,
    *,
    telemetry: Any = None,
    overview_updates: Dict[str, Any] | None = None,
    extra_entries: Sequence[Dict[str, Any]] | None = None,
    generated_by: str = "src.training.services.reporting",
) -> Dict[str, Any]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    guided_dir = root / "guided"
    base_dir = root

    overview, catalog_entries = _prepare_guided_catalog(
        root=root,
        base_dir=base_dir,
        overview_loader=_load_training_overview,
        entry_finder=_find_training_entries,
        overview_updates=overview_updates,
        extra_entries=extra_entries,
        generated_by=generated_by,
    )
    catalog_payload, overview_payload = _build_guided_payloads(
        base_dir=base_dir,
        catalog_kind="training",
        overview_kind="training",
        overview_schema_version="v1_guided_run_overview",
        overview=overview,
        overview_filename="guided/01_run_overview.json",
        primary_files={
            "adapter_behavioral_acceptance": "adapter_behavioral_acceptance.json",
            "training_summary": "training/summary.json",
            "production_readiness": "production_readiness.json",
            "test_metric_gate": "test/metric_gate.json",
        },
        catalog_entries=catalog_entries,
    )

    start_here = _render_start_here(TRAINING_START_HERE_LINES, base_dir=base_dir)
    category_to_doc = dict(TRAINING_CATEGORY_DOCS)

    written_paths = _write_guided_bundle(
        guided_dir=guided_dir,
        overview_filename="guided/01_run_overview.json",
        overview_payload=overview_payload,
        catalog_payload=catalog_payload,
        start_here=start_here,
        category_to_doc=category_to_doc,
        catalog_entries=catalog_entries,
    )
    _merge_guided_telemetry(
        telemetry=telemetry,
        artifact_root=root,
        written_paths=written_paths,
        catalog_entries=catalog_entries,
        overview_payload=overview_payload,
        overview_key="run_overview",
        overview_filename="guided/01_run_overview.json",
    )
    return {
        "catalog": catalog_payload,
        "overview": overview_payload,
        "paths": {path.name: str(path) for path in written_paths},
    }


def refresh_prep_guided_artifacts(
    artifact_root: str | Path,
    *,
    telemetry: Any = None,
    overview_updates: Dict[str, Any] | None = None,
    extra_entries: Sequence[Dict[str, Any]] | None = None,
    generated_by: str = "scripts.prepare_grouped_runtime_dataset",
) -> Dict[str, Any]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    guided_dir = root / "guided"
    base_dir = root

    overview, catalog_entries = _prepare_guided_catalog(
        root=root,
        base_dir=base_dir,
        overview_loader=_load_prep_overview,
        entry_finder=_find_prep_entries,
        overview_updates=overview_updates,
        extra_entries=extra_entries,
        generated_by=generated_by,
    )
    catalog_payload, overview_payload = _build_guided_payloads(
        base_dir=base_dir,
        catalog_kind="data_prep",
        overview_kind="data_prep",
        overview_schema_version="v1_guided_prep_overview",
        overview=overview,
        overview_filename="guided/01_prep_overview.json",
        primary_files={
            "prep_summary": "prep_summary.json",
            "split_manifest": "proposed_split_manifest.json",
        },
        catalog_entries=catalog_entries,
    )
    start_here = _render_start_here(PREP_START_HERE_LINES, base_dir=base_dir)
    category_to_doc = dict(PREP_CATEGORY_DOCS)

    written_paths = _write_guided_bundle(
        guided_dir=guided_dir,
        overview_filename="guided/01_prep_overview.json",
        overview_payload=overview_payload,
        catalog_payload=catalog_payload,
        start_here=start_here,
        category_to_doc=category_to_doc,
        catalog_entries=catalog_entries,
    )
    _merge_guided_telemetry(
        telemetry=telemetry,
        artifact_root=root,
        written_paths=written_paths,
        catalog_entries=catalog_entries,
        overview_payload=overview_payload,
        overview_key="prep_overview",
        overview_filename="guided/01_prep_overview.json",
    )
    return {
        "catalog": catalog_payload,
        "overview": overview_payload,
        "paths": {path.name: str(path) for path in written_paths},
    }
