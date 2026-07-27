#!/usr/bin/env python3
"""Run the M2 demo checklist through the Notebook 8 helper path."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_auto_router_adapter_prediction import (  # noqa: E402
    resolve_router_adapter_handoff,
    router_handoff_skip_result,
    run_auto_router_adapter_prediction,
)
from scripts.colab_router_adapter_inference import run_inference as run_router_inference  # noqa: E402
from scripts.colab_router_adapter_inference import run_inference_batch as run_router_inference_batch  # noqa: E402
from scripts.demo_checklist_cli import build_parser as build_parser  # noqa: E402
from scripts.demo_checklist_handoff_cache import (  # noqa: E402
    _cached_handoff_by_key,
    _load_handoff_cache,
    _lookup_cached_handoff_by_key,
    _write_handoff_cache,
)
from scripts.demo_checklist_reporting import (  # noqa: E402
    build_analysis_summary,
    summarize_results,
    write_run_artifacts,
)
from scripts.demo_checklist_reporting import (  # noqa: E402
    write_markdown_report as write_markdown_report,
)
from scripts.demo_checklist_rows import (  # noqa: E402
    ChecklistRow,
    _blocked_classless_probe_handoff,
    _blocked_expected_negative_handoff,
    _classless_probe_handoff_mismatch,
    _classless_supported_probe,
    _expected_negative_target,
    _format_output_row,
    _handoff_adapter_target,
    _handoff_cache_key,
    _is_cuda_oom,
    _release_cuda_memory,
    _target_parts,
    format_elapsed_seconds,
    parse_checklist_rows,
    parse_manifest_rows,
    resolve_image_path,
    resolve_prototype_thresholds_from_calibration,
)
from scripts.demo_checklist_rows import _path_fingerprint as _path_fingerprint  # noqa: E402,F401
from scripts.demo_checklist_rows import classify_failure as classify_failure  # noqa: E402,F401
from scripts.demo_checklist_rows import evaluate_pass as evaluate_pass  # noqa: E402,F401
from src.data.transforms import preprocess_image  # noqa: E402
from src.pipeline.inference_payloads import build_router_skipped_analysis, build_success_result  # noqa: E402
from src.workflows.inference import InferenceWorkflow  # noqa: E402


def _cached_handoff(
    *,
    cache: dict[str, Any] | None,
    row: ChecklistRow,
    image_path: Path,
    router_result: dict[str, Any],
    config_env: str,
    device: str,
    enable_prototype_reconciler: bool,
    prototype_bank_path: Path | None,
    taxonomy_registry_path: Path | None,
    prototype_min_similarity: float | None,
    prototype_min_margin: float | None,
    prototype_min_negative_gap: float | None,
    prototype_target_policies: dict[str, Any] | None,
    expected_target_id: str | None = None,
    expected_class_label: str | None = None,
) -> dict[str, Any]:
    key = _handoff_cache_key(
        row=row,
        image_path=image_path,
        config_env=config_env,
        device=device,
        enable_prototype_reconciler=enable_prototype_reconciler,
        prototype_bank_path=prototype_bank_path,
        taxonomy_registry_path=taxonomy_registry_path,
        prototype_min_similarity=prototype_min_similarity,
        prototype_min_margin=prototype_min_margin,
        prototype_min_negative_gap=prototype_min_negative_gap,
        prototype_target_policies=prototype_target_policies,
        expected_target_id=expected_target_id,
        expected_class_label=expected_class_label,
    )
    return _cached_handoff_by_key(
        cache=cache,
        key=key,
        row=row,
        image_path=image_path,
        resolver=resolve_router_adapter_handoff,
        resolver_kwargs={
            "router_result": router_result,
            "enable_prototype_reconciler": enable_prototype_reconciler,
            "prototype_bank_path": prototype_bank_path,
            "taxonomy_registry_path": taxonomy_registry_path,
            "prototype_min_similarity": prototype_min_similarity,
            "prototype_min_margin": prototype_min_margin,
            "prototype_min_negative_gap": prototype_min_negative_gap,
            "prototype_target_policies": prototype_target_policies,
            "expected_target_id": expected_target_id,
            "expected_class_label": expected_class_label,
        },
    )


def _lookup_cached_handoff(
    *,
    cache: dict[str, Any] | None,
    row: ChecklistRow,
    image_path: Path,
    config_env: str,
    device: str,
    enable_prototype_reconciler: bool,
    prototype_bank_path: Path | None,
    taxonomy_registry_path: Path | None,
    prototype_min_similarity: float | None,
    prototype_min_margin: float | None,
    prototype_min_negative_gap: float | None,
    prototype_target_policies: dict[str, Any] | None,
    expected_target_id: str | None = None,
    expected_class_label: str | None = None,
) -> dict[str, Any] | None:
    key = _handoff_cache_key(
        row=row,
        image_path=image_path,
        config_env=config_env,
        device=device,
        enable_prototype_reconciler=enable_prototype_reconciler,
        prototype_bank_path=prototype_bank_path,
        taxonomy_registry_path=taxonomy_registry_path,
        prototype_min_similarity=prototype_min_similarity,
        prototype_min_margin=prototype_min_margin,
        prototype_min_negative_gap=prototype_min_negative_gap,
        prototype_target_policies=prototype_target_policies,
        expected_target_id=expected_target_id,
        expected_class_label=expected_class_label,
    )
    return _lookup_cached_handoff_by_key(cache=cache, key=key)


def _run_row(
    row: ChecklistRow,
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    mode: str,
    enable_prototype_reconciler: bool = False,
    prototype_bank_path: Path | None = None,
    taxonomy_registry_path: Path | None = None,
    prototype_min_similarity: float | None = None,
    prototype_min_margin: float | None = None,
    prototype_min_negative_gap: float | None = None,
    prototype_target_policies: dict[str, Any] | None = None,
    router_result_override: dict[str, Any] | None = None,
    handoff_result_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path, asset_status = resolve_image_path(row.source, repo_root)
    if asset_status != "ok" or image_path is None:
        result: dict[str, Any] = {
            "status": "asset_missing",
            "crop": None,
            "part": None,
            "diagnosis": None,
            "confidence": 0.0,
            "message": f"Checklist source could not be resolved: {row.source}",
        }
    else:
        crop_hint, part_hint = _target_parts(row.expected_target)
        try:
            if mode == "asset-audit":
                result = {
                    "status": "asset_ready",
                    "crop": crop_hint,
                    "part": part_hint,
                    "diagnosis": None,
                    "confidence": 0.0,
                    "message": "Asset resolved; inference not run in asset-audit mode.",
                }
            elif mode == "adapter-smoke":
                if crop_hint is None or part_hint is None:
                    result = {
                        "status": "router_skipped_target_not_adapter_eligible",
                        "crop": crop_hint,
                        "part": part_hint,
                        "diagnosis": None,
                        "confidence": 0.0,
                        "message": "Expected target is not adapter-eligible for trusted-hint smoke.",
                    }
                else:
                    workflow = InferenceWorkflow(
                        environment=config_env,
                        device=device,
                        adapter_root=adapter_root,
                    )
                    result = workflow.predict(
                        image_path,
                        crop_hint=crop_hint,
                        part_hint=part_hint,
                        return_ood=True,
                        trust_crop_hint=True,
                    )
            else:
                router_result = (
                    router_result_override
                    if router_result_override is not None
                    else run_router_inference(
                        image_path,
                        config_env=config_env,
                        device=device,
                    )
                )
                if _expected_negative_target(row.expected_target, row.expected_behavior):
                    handoff_result = handoff_result_override or resolve_router_adapter_handoff(
                        image_path,
                        router_result=router_result,
                        enable_prototype_reconciler=enable_prototype_reconciler,
                        prototype_bank_path=prototype_bank_path,
                        taxonomy_registry_path=taxonomy_registry_path,
                        prototype_min_similarity=prototype_min_similarity,
                        prototype_min_margin=prototype_min_margin,
                        prototype_min_negative_gap=prototype_min_negative_gap,
                        prototype_target_policies=prototype_target_policies,
                        expected_target_id=row.expected_target,
                        expected_class_label=row.expected_class,
                    )
                    result = router_handoff_skip_result(_blocked_expected_negative_handoff(row, handoff_result))
                else:
                    handoff_result = handoff_result_override
                    if _classless_supported_probe(row):
                        handoff_result = handoff_result or resolve_router_adapter_handoff(
                            image_path,
                            router_result=router_result,
                            enable_prototype_reconciler=enable_prototype_reconciler,
                            prototype_bank_path=prototype_bank_path,
                            taxonomy_registry_path=taxonomy_registry_path,
                            prototype_min_similarity=prototype_min_similarity,
                            prototype_min_margin=prototype_min_margin,
                            prototype_min_negative_gap=prototype_min_negative_gap,
                            prototype_target_policies=prototype_target_policies,
                            expected_target_id=row.expected_target,
                            expected_class_label=row.expected_class,
                        )
                        if _classless_probe_handoff_mismatch(row, handoff_result):
                            result = router_handoff_skip_result(_blocked_classless_probe_handoff(row, handoff_result))
                            return _format_output_row(
                                row,
                                image_path=image_path,
                                result=result,
                                asset_status=asset_status,
                                mode=mode,
                            )
                    result = run_auto_router_adapter_prediction(
                        image_path,
                        router_result=router_result,
                        config_env=config_env,
                        device=device,
                        adapter_root=adapter_root,
                        return_ood=True,
                        enable_prototype_reconciler=enable_prototype_reconciler,
                        prototype_bank_path=prototype_bank_path,
                        taxonomy_registry_path=taxonomy_registry_path,
                        prototype_min_similarity=prototype_min_similarity,
                        prototype_min_margin=prototype_min_margin,
                        prototype_min_negative_gap=prototype_min_negative_gap,
                        prototype_target_policies=prototype_target_policies,
                        expected_target_id=row.expected_target,
                        expected_class_label=row.expected_class,
                        handoff_result=handoff_result,
                    )
        except Exception as exc:  # Notebook execution surfaces dependency failures as runtime exceptions.
            if _is_cuda_oom(exc):
                _release_cuda_memory(device)
            result = {
                "status": "router_unavailable",
                "crop": None,
                "part": None,
                "diagnosis": None,
                "confidence": 0.0,
                "message": str(exc),
            }

    return _format_output_row(row, image_path=image_path, result=result, asset_status=asset_status, mode=mode)


def _run_adapter_batched_rows(
    chunk: list[tuple[ChecklistRow, Path | None, str]],
    handoffs: list[dict[str, Any]],
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    adapter_batch_size: int,
    workflow: Any | None = None,
) -> list[dict[str, Any]] | None:
    if adapter_batch_size <= 1:
        return None
    if workflow is None:
        workflow = InferenceWorkflow(
            environment=config_env,
            device=device,
            adapter_root=adapter_root,
        )
    if workflow.runtime.input_guard_enabled:
        return None

    output_rows: list[dict[str, Any] | None] = [None] * len(chunk)
    grouped: dict[tuple[str, str], list[tuple[int, ChecklistRow, Path, dict[str, Any]]]] = {}
    for index, ((row, image_path, asset_status), handoff) in enumerate(zip(chunk, handoffs)):
        if image_path is None or asset_status != "ok":
            return None
        if _expected_negative_target(row.expected_target, row.expected_behavior):
            output_rows[index] = _format_output_row(
                row,
                image_path=image_path,
                result=router_handoff_skip_result(_blocked_expected_negative_handoff(row, handoff)),
                asset_status="ok",
                mode="official",
            )
            continue
        if _classless_probe_handoff_mismatch(row, handoff):
            output_rows[index] = _format_output_row(
                row,
                image_path=image_path,
                result=router_handoff_skip_result(_blocked_classless_probe_handoff(row, handoff)),
                asset_status="ok",
                mode="official",
            )
            continue
        target = _handoff_adapter_target(handoff)
        if target is None:
            output_rows[index] = _format_output_row(
                row,
                image_path=image_path,
                result=router_handoff_skip_result(handoff),
                asset_status="ok",
                mode="official",
            )
            continue
        grouped.setdefault(target, []).append((index, row, image_path, handoff))

    for (crop, part), items in grouped.items():
        try:
            adapter = workflow.runtime.load_adapter(crop, part_name=part)
            for start in range(0, len(items), max(1, int(adapter_batch_size))):
                batch_items = items[start : start + max(1, int(adapter_batch_size))]
                image_tensors = [
                    preprocess_image(
                        workflow.runtime._coerce_image(image_path),
                        target_size=workflow.runtime.target_size,
                    )
                    for _, _, image_path, _ in batch_items
                ]
                adapter_results = adapter.predict_batch_with_ood(torch.stack(image_tensors, dim=0))
                if len(adapter_results) != len(batch_items):
                    raise RuntimeError(
                        f"Batch adapter returned {len(adapter_results)} results for {len(batch_items)} rows."
                    )
                for (index, row, image_path, handoff), adapter_result in zip(batch_items, adapter_results):
                    router_analysis = build_router_skipped_analysis(
                        crop_name=crop,
                        part_name=part,
                        router_confidence=1.0,
                        status="trusted_hint_skipped",
                        message="Router skipped because trust_crop_hint=True.",
                    )
                    payload = build_success_result(
                        crop_name=crop,
                        part_name=part,
                        router_confidence=1.0,
                        result=adapter_result,
                        include_ood=True,
                        router_analysis=router_analysis,
                    ).to_dict(include_ood=True)
                    payload["router_source"] = dict(handoff.get("router") or {})
                    payload["router_handoff"] = {
                        "adapter_ran": True,
                        "source_status": str(handoff.get("status") or ""),
                        "crop": crop,
                        "part": part,
                        "prototype_reconciliation": dict(handoff.get("prototype_reconciliation") or {}),
                    }
                    output_rows[index] = _format_output_row(
                        row,
                        image_path=image_path,
                        result=payload,
                        asset_status="ok",
                        mode="official",
                    )
        except Exception as exc:
            if _is_cuda_oom(exc):
                _release_cuda_memory(device)
                print(
                    f"[M2] Adapter CUDA OOM at adapter_batch_size={adapter_batch_size}; "
                    "falling back to per-row adapter inference.",
                    file=sys.stderr,
                )
            return None

    return [row for row in output_rows if row is not None]


def _row_from_batch_router(
    row: ChecklistRow,
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    enable_prototype_reconciler: bool = False,
    prototype_bank_path: Path | None = None,
    taxonomy_registry_path: Path | None = None,
    prototype_min_similarity: float | None = None,
    prototype_min_margin: float | None = None,
    prototype_min_negative_gap: float | None = None,
    prototype_target_policies: dict[str, Any] | None = None,
    router_result: dict[str, Any],
    handoff_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _run_row(
        row,
        repo_root=repo_root,
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        mode="official",
        enable_prototype_reconciler=enable_prototype_reconciler,
        prototype_bank_path=prototype_bank_path,
        taxonomy_registry_path=taxonomy_registry_path,
        prototype_min_similarity=prototype_min_similarity,
        prototype_min_margin=prototype_min_margin,
        prototype_min_negative_gap=prototype_min_negative_gap,
        prototype_target_policies=prototype_target_policies,
        router_result_override=router_result,
        handoff_result_override=handoff_result,
    )


def _run_official_batch_rows(
    rows: list[ChecklistRow],
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    batch_size: int,
    adapter_batch_size: int = 1,
    enable_prototype_reconciler: bool = False,
    prototype_bank_path: Path | None = None,
    taxonomy_registry_path: Path | None = None,
    prototype_min_similarity: float | None = None,
    prototype_min_margin: float | None = None,
    prototype_min_negative_gap: float | None = None,
    prototype_target_policies: dict[str, Any] | None = None,
    handoff_cache: dict[str, Any] | None = None,
    stop_on_dependency_blocker: bool = False,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    resolved: list[tuple[ChecklistRow, Path | None, str]] = [
        (row, *resolve_image_path(row.source, repo_root)) for row in rows
    ]
    chunk_size = max(1, int(batch_size))
    adapter_batch_workflow = (
        InferenceWorkflow(environment=config_env, device=device, adapter_root=adapter_root)
        if adapter_batch_size > 1
        else None
    )

    def _maybe_stop_on_blocker() -> bool:
        if not stop_on_dependency_blocker:
            return False
        blocker_index = next(
            (
                index
                for index, row in enumerate(output_rows)
                if row.get("failure_bucket") in {"dependency_access", "cuda_oom"}
            ),
            None,
        )
        if blocker_index is None:
            return False
        del output_rows[blocker_index + 1 :]
        return True

    start = 0
    while start < len(resolved):
        chunk = resolved[start : start + chunk_size]
        if any(asset_status != "ok" or image_path is None for _, image_path, asset_status in chunk):
            for row, _, _ in chunk:
                output_rows.append(
                    _run_row(
                        row,
                        repo_root=repo_root,
                        config_env=config_env,
                        device=device,
                        adapter_root=adapter_root,
                        mode="official",
                        enable_prototype_reconciler=enable_prototype_reconciler,
                        prototype_bank_path=prototype_bank_path,
                        taxonomy_registry_path=taxonomy_registry_path,
                        prototype_min_similarity=prototype_min_similarity,
                        prototype_min_margin=prototype_min_margin,
                        prototype_min_negative_gap=prototype_min_negative_gap,
                        prototype_target_policies=prototype_target_policies,
                    )
                )
            start += len(chunk)
            if _maybe_stop_on_blocker():
                return output_rows
            continue
        image_paths = [image_path for _, image_path, _ in chunk if image_path is not None]
        handoffs = [
            _lookup_cached_handoff(
                cache=handoff_cache,
                row=row,
                image_path=image_path,
                config_env=config_env,
                device=device,
                enable_prototype_reconciler=enable_prototype_reconciler,
                prototype_bank_path=prototype_bank_path,
                taxonomy_registry_path=taxonomy_registry_path,
                prototype_min_similarity=prototype_min_similarity,
                prototype_min_margin=prototype_min_margin,
                prototype_min_negative_gap=prototype_min_negative_gap,
                prototype_target_policies=prototype_target_policies,
                expected_target_id=row.expected_target,
                expected_class_label=row.expected_class,
            )
            for row, image_path, _ in chunk
            if image_path is not None
        ]
        router_results: list[dict[str, Any]] = [{} for _ in chunk]
        if len(handoffs) != len(chunk) or any(handoff is None for handoff in handoffs):
            try:
                router_results = run_router_inference_batch(
                    image_paths,
                    config_env=config_env,
                    device=device,
                )
                if len(router_results) != len(chunk):
                    raise RuntimeError(f"Batch router returned {len(router_results)} results for {len(chunk)} rows.")
            except Exception as exc:
                if _is_cuda_oom(exc) and chunk_size > 1:
                    next_chunk_size = max(1, chunk_size // 2)
                    print(
                        f"[M2] Router CUDA OOM at batch_size={chunk_size}; "
                        f"retrying from row {start + 1} with batch_size={next_chunk_size}.",
                        file=sys.stderr,
                    )
                    _release_cuda_memory(device)
                    chunk_size = next_chunk_size
                    continue
                if _is_cuda_oom(exc):
                    _release_cuda_memory(device)
                for row, _, _ in chunk:
                    output_rows.append(
                        _run_row(
                            row,
                            repo_root=repo_root,
                            config_env=config_env,
                            device=device,
                            adapter_root=adapter_root,
                            mode="official",
                            enable_prototype_reconciler=enable_prototype_reconciler,
                            prototype_bank_path=prototype_bank_path,
                            taxonomy_registry_path=taxonomy_registry_path,
                            prototype_min_similarity=prototype_min_similarity,
                            prototype_min_margin=prototype_min_margin,
                            prototype_min_negative_gap=prototype_min_negative_gap,
                            prototype_target_policies=prototype_target_policies,
                        )
                    )
                start += len(chunk)
                if _maybe_stop_on_blocker():
                    return output_rows
                continue
            handoffs = [
                _cached_handoff(
                    cache=handoff_cache,
                    row=row,
                    image_path=image_path,
                    router_result=router_result,
                    config_env=config_env,
                    device=device,
                    enable_prototype_reconciler=enable_prototype_reconciler,
                    prototype_bank_path=prototype_bank_path,
                    taxonomy_registry_path=taxonomy_registry_path,
                    prototype_min_similarity=prototype_min_similarity,
                    prototype_min_margin=prototype_min_margin,
                    prototype_min_negative_gap=prototype_min_negative_gap,
                    prototype_target_policies=prototype_target_policies,
                    expected_target_id=row.expected_target,
                    expected_class_label=row.expected_class,
                )
                for (row, image_path, _), router_result in zip(chunk, router_results)
                if image_path is not None
            ]
            if len(handoffs) != len(chunk):
                raise RuntimeError(f"Resolved {len(handoffs)} router/prototype handoffs for {len(chunk)} rows.")
        batched_adapter_rows = _run_adapter_batched_rows(
            chunk,
            [dict(handoff or {}) for handoff in handoffs],
            repo_root=repo_root,
            config_env=config_env,
            device=device,
            adapter_root=adapter_root,
            adapter_batch_size=adapter_batch_size,
            workflow=adapter_batch_workflow,
        )
        if batched_adapter_rows is not None:
            output_rows.extend(batched_adapter_rows)
            start += len(chunk)
            if _maybe_stop_on_blocker():
                return output_rows
            continue
        for (row, _, _), router_result, handoff in zip(chunk, router_results, handoffs):
            output_rows.append(
                _row_from_batch_router(
                    row,
                    repo_root=repo_root,
                    config_env=config_env,
                    device=device,
                    adapter_root=adapter_root,
                    enable_prototype_reconciler=enable_prototype_reconciler,
                    prototype_bank_path=prototype_bank_path,
                    taxonomy_registry_path=taxonomy_registry_path,
                    prototype_min_similarity=prototype_min_similarity,
                    prototype_min_margin=prototype_min_margin,
                    prototype_min_negative_gap=prototype_min_negative_gap,
                    prototype_target_policies=prototype_target_policies,
                    router_result=router_result,
                    handoff_result=dict(handoff or {}),
                )
            )
        start += len(chunk)

    return output_rows


def main() -> int:
    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    rows = [] if args.no_checklist else parse_checklist_rows(args.checklist)
    for manifest_path in args.extra_manifest:
        rows.extend(parse_manifest_rows(manifest_path))
    if args.only_local:
        rows = [row for row in rows if row.source.startswith("local_test_pool:")]
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    mode = "adapter-smoke" if args.trust_expected_target else str(args.mode)
    (
        prototype_min_similarity,
        prototype_min_margin,
        prototype_min_negative_gap,
        calibration_report,
        prototype_target_policies,
    ) = resolve_prototype_thresholds_from_calibration(
        args.prototype_calibration_report,
        min_similarity=args.prototype_min_similarity,
        min_margin=args.prototype_min_margin,
        min_negative_gap=args.prototype_min_negative_gap,
    )

    batch_size = max(1, int(args.batch_size or 1))
    adapter_batch_size = max(1, int(args.adapter_batch_size or 1))
    handoff_cache_path = args.handoff_cache if mode == "official" and batch_size > 1 else None
    handoff_cache = _load_handoff_cache(handoff_cache_path, refresh=bool(args.refresh_handoff_cache))
    if mode == "official" and batch_size > 1:
        output_rows = _run_official_batch_rows(
            rows,
            repo_root=repo_root,
            config_env=str(args.config_env),
            device=str(args.device),
            adapter_root=args.adapter_root,
            batch_size=batch_size,
            adapter_batch_size=adapter_batch_size,
            enable_prototype_reconciler=bool(args.enable_prototype_reconciler),
            prototype_bank_path=args.prototype_bank,
            taxonomy_registry_path=args.taxonomy_registry,
            prototype_min_similarity=prototype_min_similarity,
            prototype_min_margin=prototype_min_margin,
            prototype_min_negative_gap=prototype_min_negative_gap,
            prototype_target_policies=prototype_target_policies,
            handoff_cache=handoff_cache,
            stop_on_dependency_blocker=bool(args.stop_on_dependency_blocker),
        )
        _write_handoff_cache(handoff_cache_path, handoff_cache)
        if args.stop_on_dependency_blocker:
            blocker_index = next(
                (
                    index
                    for index, row in enumerate(output_rows)
                    if row.get("failure_bucket") in {"dependency_access", "cuda_oom"}
                ),
                None,
            )
            if blocker_index is not None:
                output_rows = output_rows[: blocker_index + 1]
    else:
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            result = _run_row(
                row,
                repo_root=repo_root,
                config_env=str(args.config_env),
                device=str(args.device),
                adapter_root=args.adapter_root,
                mode=mode,
                enable_prototype_reconciler=bool(args.enable_prototype_reconciler),
                prototype_bank_path=args.prototype_bank,
                taxonomy_registry_path=args.taxonomy_registry,
                prototype_min_similarity=prototype_min_similarity,
                prototype_min_margin=prototype_min_margin,
                prototype_min_negative_gap=prototype_min_negative_gap,
                prototype_target_policies=prototype_target_policies,
            )
            output_rows.append(result)
            if args.stop_on_dependency_blocker and result.get("failure_bucket") in {"dependency_access", "cuda_oom"}:
                break

    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = time.perf_counter() - start_perf
    report = {
        "schema_version": "v1_m2_demo_checklist_run",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_elapsed_seconds(elapsed_seconds),
        "generated_at": finished_at.isoformat(),
        "checklist": str(args.checklist),
        "device": str(args.device),
        "adapter_root": str(args.adapter_root),
        "mode": mode,
        "batch_size": batch_size,
        "adapter_batch_size": adapter_batch_size,
        "handoff_cache": {
            "enabled": handoff_cache_path is not None,
            "path": str(handoff_cache_path) if handoff_cache_path else "",
            "refresh": bool(args.refresh_handoff_cache),
            "stats": dict(handoff_cache.get("stats", {})) if isinstance(handoff_cache, dict) else {},
        },
        "prototype_reconciler": {
            "enabled": bool(args.enable_prototype_reconciler),
            "prototype_bank": str(args.prototype_bank) if args.prototype_bank else "",
            "taxonomy_registry": str(args.taxonomy_registry) if args.taxonomy_registry else "",
            "prototype_calibration_report": calibration_report,
            "prototype_min_similarity": prototype_min_similarity,
            "prototype_min_margin": prototype_min_margin,
            "prototype_min_negative_gap": prototype_min_negative_gap,
            "prototype_target_policy_count": len(prototype_target_policies),
        },
        "trust_expected_target": mode == "adapter-smoke",
        "summary": summarize_results(output_rows),
        "rows": output_rows,
    }
    analysis = build_analysis_summary(output_rows)
    report["analysis_summary"] = analysis
    write_run_artifacts(
        report,
        output_path=args.output,
        markdown_output_path=args.markdown_output,
        analysis_output_path=args.analysis_output,
        analysis_markdown_output_path=args.analysis_markdown_output,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
