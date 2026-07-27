"""Prototype-router calibration policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoredRow:
    image_id: str
    expected_target: str
    expected_behavior: str
    predicted_target: str | None
    similarity: float
    margin: float
    resolved_image: str
    status: str
    prototype_class_label: str | None = None
    prototype_level: str = "target"
    expected_class: str = ""


def _target_is_supported_positive(target: str) -> bool:
    return "__" in target and not target.startswith(("unknown_crop", "non_plant")) and not target.endswith(
        "__unknown_part"
    )


def _target_is_negative(target: str, expected_behavior: str = "") -> bool:
    normalized = str(target or "").strip().lower()
    behavior = str(expected_behavior or "").strip().lower()
    return (
        normalized in {"unknown_crop", "non_plant"}
        or normalized.endswith("__unknown_part")
        or "unsupported" in behavior
        or "abstain" in behavior
    )


def _target_part(target: str | None) -> str:
    text = str(target or "").strip().lower()
    if "__" not in text:
        return ""
    return text.rsplit("__", 1)[1]


def _supported_cross_part_wrong_rows(wrong_rows: list[ScoredRow], target_id: str) -> list[ScoredRow]:
    target_part = _target_part(target_id)
    return [
        row
        for row in wrong_rows
        if _target_is_supported_positive(row.expected_target)
        and target_part
        and _target_part(row.expected_target) != target_part
    ]


def _row_payload(row: ScoredRow) -> dict[str, Any]:
    return {
        "image_id": row.image_id,
        "expected_target": row.expected_target,
        "expected_class": row.expected_class,
        "predicted_target": row.predicted_target,
        "prototype_class_label": row.prototype_class_label,
        "prototype_level": row.prototype_level,
        "similarity": row.similarity,
        "margin": row.margin,
    }


def evaluate_thresholds(
    rows: list[ScoredRow],
    *,
    min_similarity: float,
    min_margin: float,
    min_negative_gap: float = 0.0,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.status == "ok"]
    supported = [row for row in eligible if _target_is_supported_positive(row.expected_target)]
    negatives = [row for row in eligible if _target_is_negative(row.expected_target, row.expected_behavior)]
    accepted = [
        row
        for row in eligible
        if row.predicted_target
        and row.similarity >= min_similarity
        and row.margin >= min_margin
        and row.margin >= min_negative_gap
    ]
    supported_accepted = [row for row in accepted if _target_is_supported_positive(row.expected_target)]
    correct = [row for row in supported_accepted if row.predicted_target == row.expected_target]
    wrong = [row for row in supported_accepted if row.predicted_target != row.expected_target]
    negative_false_accepts = [row for row in accepted if _target_is_negative(row.expected_target, row.expected_behavior)]
    non_plant_false_accepts = [
        row for row in negative_false_accepts if str(row.expected_target or "").strip().lower() == "non_plant"
    ]
    total = len(eligible)
    coverage = len(accepted) / total if total else 0.0
    precision = len(correct) / len(accepted) if accepted else 0.0
    accuracy = len(correct) / total if total else 0.0
    supported_coverage = len(supported_accepted) / len(supported) if supported else 0.0
    supported_precision = len(correct) / len(supported_accepted) if supported_accepted else 0.0
    negative_false_accept_rate = len(negative_false_accepts) / len(negatives) if negatives else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "min_negative_gap": min_negative_gap,
        "promotion_mode": "prototype_override",
        "eligible": total,
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong": len(wrong),
        "coverage": round(coverage, 6),
        "precision": round(precision, 6),
        "accuracy": round(accuracy, 6),
        "supported_rows": len(supported),
        "supported_accepted": len(supported_accepted),
        "supported_correct": len(correct),
        "supported_wrong": len(wrong),
        "supported_wrong_image_ids": [row.image_id for row in wrong[:25]],
        "supported_wrong_rows": [
            {
                "image_id": row.image_id,
                "expected_target": row.expected_target,
                "predicted_target": row.predicted_target,
                "prototype_class_label": row.prototype_class_label,
                "prototype_level": row.prototype_level,
                "similarity": row.similarity,
                "margin": row.margin,
            }
            for row in wrong[:25]
        ],
        "supported_wrong_truncated": len(wrong) > 25,
        "supported_coverage": round(supported_coverage, 6),
        "supported_precision": round(supported_precision, 6),
        "negative_rows": len(negatives),
        "negative_false_accept_count": len(negative_false_accepts),
        "negative_false_accept_rate": round(negative_false_accept_rate, 6),
        "non_plant_false_accept_count": len(non_plant_false_accepts),
    }


def evaluate_target_thresholds(
    rows: list[ScoredRow],
    *,
    target_id: str,
    min_similarity: float,
    min_margin: float,
    min_negative_gap: float = 0.0,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.status == "ok"]
    target_rows = [row for row in eligible if row.expected_target == target_id]
    accepted = [
        row
        for row in eligible
        if row.predicted_target == target_id
        and row.similarity >= min_similarity
        and row.margin >= min_margin
        and row.margin >= min_negative_gap
    ]
    supported_accepted = [row for row in accepted if _target_is_supported_positive(row.expected_target)]
    correct = [row for row in supported_accepted if row.expected_target == target_id]
    wrong = [row for row in supported_accepted if row.expected_target != target_id]
    cross_part_wrong = _supported_cross_part_wrong_rows(wrong, target_id)
    negative_false_accepts = [row for row in accepted if _target_is_negative(row.expected_target, row.expected_behavior)]
    non_plant_false_accepts = [
        row for row in negative_false_accepts if str(row.expected_target or "").strip().lower() == "non_plant"
    ]
    supported_precision = len(correct) / len(supported_accepted) if supported_accepted else 0.0
    target_coverage = len(correct) / len(target_rows) if target_rows else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "min_negative_gap": min_negative_gap,
        "promotion_mode": "prototype_override",
        "target_id": target_id,
        "eligible": len(eligible),
        "target_rows": len(target_rows),
        "accepted": len(accepted),
        "supported_accepted": len(supported_accepted),
        "supported_correct": len(correct),
        "supported_wrong": len(wrong),
        "supported_precision": round(supported_precision, 6),
        "target_coverage": round(target_coverage, 6),
        "supported_cross_part_wrong": len(cross_part_wrong),
        "supported_cross_part_wrong_image_ids": [row.image_id for row in cross_part_wrong[:25]],
        "supported_cross_part_wrong_rows": [_row_payload(row) for row in cross_part_wrong[:25]],
        "supported_cross_part_wrong_truncated": len(cross_part_wrong) > 25,
        "negative_false_accept_count": len(negative_false_accepts),
        "non_plant_false_accept_count": len(non_plant_false_accepts),
        "supported_wrong_image_ids": [row.image_id for row in wrong[:25]],
        "supported_wrong_rows": [_row_payload(row) for row in wrong[:25]],
        "supported_wrong_truncated": len(wrong) > 25,
    }


def evaluate_class_thresholds(
    rows: list[ScoredRow],
    *,
    target_id: str,
    class_label: str,
    min_similarity: float,
    min_margin: float,
    min_negative_gap: float = 0.0,
    include_negative_rows: bool = True,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.status == "ok"
        and (include_negative_rows or not _target_is_negative(row.expected_target, row.expected_behavior))
    ]
    target_rows = [row for row in eligible if row.expected_target == target_id]
    accepted = [
        row
        for row in eligible
        if row.predicted_target == target_id
        and row.prototype_class_label == class_label
        and row.similarity >= min_similarity
        and row.margin >= min_margin
        and row.margin >= min_negative_gap
    ]
    supported_accepted = [row for row in accepted if _target_is_supported_positive(row.expected_target)]
    correct = [row for row in supported_accepted if row.expected_target == target_id]
    wrong = [row for row in supported_accepted if row.expected_target != target_id]
    exact_class_correct = [
        row
        for row in supported_accepted
        if row.expected_target == target_id and str(row.expected_class or "").strip() == class_label
    ]
    exact_class_wrong = [row for row in supported_accepted if row not in exact_class_correct]
    cross_part_wrong = _supported_cross_part_wrong_rows(wrong, target_id)
    negative_false_accepts = [row for row in accepted if _target_is_negative(row.expected_target, row.expected_behavior)]
    non_plant_false_accepts = [
        row for row in negative_false_accepts if str(row.expected_target or "").strip().lower() == "non_plant"
    ]
    supported_precision = len(correct) / len(supported_accepted) if supported_accepted else 0.0
    target_coverage = len(correct) / len(target_rows) if target_rows else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "min_negative_gap": min_negative_gap,
        "promotion_mode": "prototype_override",
        "target_id": target_id,
        "class_label": class_label,
        "eligible": len(eligible),
        "target_rows": len(target_rows),
        "accepted": len(accepted),
        "supported_accepted": len(supported_accepted),
        "supported_correct": len(correct),
        "supported_wrong": len(wrong),
        "exact_class_supported_correct": len(exact_class_correct),
        "exact_class_supported_wrong": len(exact_class_wrong),
        "supported_precision": round(supported_precision, 6),
        "target_coverage": round(target_coverage, 6),
        "supported_cross_part_wrong": len(cross_part_wrong),
        "supported_cross_part_wrong_image_ids": [row.image_id for row in cross_part_wrong[:25]],
        "supported_cross_part_wrong_rows": [_row_payload(row) for row in cross_part_wrong[:25]],
        "supported_cross_part_wrong_truncated": len(cross_part_wrong) > 25,
        "negative_false_accept_count": len(negative_false_accepts),
        "non_plant_false_accept_count": len(non_plant_false_accepts),
        "supported_wrong_image_ids": [row.image_id for row in wrong[:25]],
        "supported_wrong_rows": [_row_payload(row) for row in wrong[:25]],
        "supported_wrong_truncated": len(wrong) > 25,
        "exact_class_supported_wrong_image_ids": [row.image_id for row in exact_class_wrong[:25]],
        "exact_class_supported_wrong_rows": [_row_payload(row) for row in exact_class_wrong[:25]],
        "exact_class_supported_wrong_truncated": len(exact_class_wrong) > 25,
    }


def calibrate_class_policies(
    rows: list[ScoredRow],
    *,
    target_id: str,
    similarity_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    negative_gap_grid: tuple[float, ...],
    min_precision: float,
    max_supported_wrong: int | None,
    max_cross_part_supported_wrong: int | None,
    min_accepted: int,
    require_zero_non_plant_false_accepts: bool,
    max_negative_false_accepts: int | None,
    include_negative_rows: bool = True,
) -> dict[str, Any]:
    class_labels = sorted(
        {
            str(row.prototype_class_label or "").strip()
            for row in rows
            if row.status == "ok" and row.predicted_target == target_id and str(row.prototype_class_label or "").strip()
        }
    )
    policies: dict[str, dict[str, Any]] = {}
    for class_label in class_labels:
        candidates: list[dict[str, Any]] = []
        for min_similarity in similarity_grid:
            for min_margin in margin_grid:
                for min_negative_gap in negative_gap_grid:
                    result = evaluate_class_thresholds(
                        rows,
                        target_id=target_id,
                        class_label=class_label,
                        min_similarity=min_similarity,
                        min_margin=min_margin,
                        min_negative_gap=min_negative_gap,
                        include_negative_rows=include_negative_rows,
                    )
                    result["eligible_for_promotion"] = (
                        result["supported_accepted"] >= min_accepted
                        and result["supported_precision"] >= min_precision
                        and (max_supported_wrong is None or result["supported_wrong"] <= max_supported_wrong)
                        and (
                            max_cross_part_supported_wrong is None
                            or result["supported_cross_part_wrong"] <= max_cross_part_supported_wrong
                        )
                        and (
                            not require_zero_non_plant_false_accepts
                            or result["non_plant_false_accept_count"] == 0
                        )
                        and (
                            max_negative_false_accepts is None
                            or result["negative_false_accept_count"] <= max_negative_false_accepts
                        )
                    )
                    candidates.append(result)
        candidates.sort(
            key=lambda item: (
                bool(item["eligible_for_promotion"]),
                float(item["supported_correct"]),
                float(item["target_coverage"]),
                -float(item["negative_false_accept_count"]),
                float(item["supported_precision"]),
                -float(item["min_similarity"]),
                -float(item["min_margin"]),
                -float(item["min_negative_gap"]),
            ),
            reverse=True,
        )
        selected = candidates[0] if candidates and candidates[0].get("eligible_for_promotion") else None
        if selected and _target_part(target_id) == "fruit" and int(selected.get("supported_cross_part_wrong") or 0) == 0:
            selected = {**selected, "allow_part_conflict_override": True}
        exact_rescue_min_accepted = min(3, int(min_accepted))
        exact_rescue_candidates = [
            item
            for item in candidates
            if int(item.get("exact_class_supported_correct") or 0) >= exact_rescue_min_accepted
            and int(item.get("exact_class_supported_wrong") or 0) == 0
            and int(item.get("supported_cross_part_wrong") or 0) == 0
            and int(item.get("negative_false_accept_count") or 0) <= int(max_negative_false_accepts or 0)
            and (
                not require_zero_non_plant_false_accepts
                or int(item.get("non_plant_false_accept_count") or 0) == 0
            )
        ]
        exact_rescue_candidates.sort(
            key=lambda item: (
                float(item["exact_class_supported_correct"]),
                float(item["target_coverage"]),
                -float(item["min_similarity"]),
                -float(item["min_margin"]),
                -float(item["min_negative_gap"]),
            ),
            reverse=True,
        )
        exact_rescue = exact_rescue_candidates[0] if exact_rescue_candidates else None
        if exact_rescue:
            exact_rescue = {
                **exact_rescue,
                "allow_expected_class_rescue": True,
                "ignore_hard_negative_gap": True,
                "exact_class_rescue_min_accepted": exact_rescue_min_accepted,
            }
        best_candidate = candidates[0] if candidates else None
        failure_reasons: list[str] = []
        if not selected and best_candidate:
            if int(best_candidate.get("supported_accepted") or 0) < int(min_accepted):
                failure_reasons.append("supported_accepted_below_class_min")
            if float(best_candidate.get("supported_precision") or 0.0) < float(min_precision):
                failure_reasons.append("supported_precision_below_class_target")
            if max_supported_wrong is not None and int(best_candidate.get("supported_wrong") or 0) > int(
                max_supported_wrong
            ):
                failure_reasons.append("supported_wrong_above_class_target")
            if (
                max_cross_part_supported_wrong is not None
                and int(best_candidate.get("supported_cross_part_wrong") or 0) > int(max_cross_part_supported_wrong)
            ):
                failure_reasons.append("supported_cross_part_wrong_above_class_target")
            if (
                max_negative_false_accepts is not None
                and int(best_candidate.get("negative_false_accept_count") or 0) > int(max_negative_false_accepts)
            ):
                failure_reasons.append("negative_false_accepts_above_class_target")
            if require_zero_non_plant_false_accepts and int(best_candidate.get("non_plant_false_accept_count") or 0):
                failure_reasons.append("non_plant_false_accepts_present")
        policies[class_label] = {
            "status": "class_specific" if selected else "no_eligible_policy",
            "selected_policy": selected,
            "exact_class_rescue_policy": exact_rescue,
            "best_candidate": best_candidate,
            "failure_reasons": failure_reasons,
        }
    return policies


def calibrate(
    rows: list[ScoredRow],
    *,
    similarity_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    negative_gap_grid: tuple[float, ...] = (0.0,),
    min_precision: float,
    min_coverage: float,
    require_zero_non_plant_false_accepts: bool = True,
    max_negative_false_accepts: int | None = 0,
    max_negative_false_accept_rate: float | None = 0.05,
    max_supported_wrong: int | None = None,
    include_target_policies: bool = True,
    target_policy_negative_mode: str = "all",
    target_min_precision: float | None = None,
    target_max_supported_wrong: int | None = None,
    target_max_cross_part_supported_wrong: int | None = 0,
    include_class_policies: bool = True,
    target_class_min_accepted: int = 5,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for min_similarity in similarity_grid:
        for min_margin in margin_grid:
            for min_negative_gap in negative_gap_grid:
                result = evaluate_thresholds(
                    rows,
                    min_similarity=min_similarity,
                    min_margin=min_margin,
                    min_negative_gap=min_negative_gap,
                )
                result["eligible_for_promotion"] = (
                    result["supported_precision"] >= min_precision
                    and result["supported_coverage"] >= min_coverage
                    and (
                        not require_zero_non_plant_false_accepts
                        or result["non_plant_false_accept_count"] == 0
                    )
                    and (
                        max_negative_false_accepts is None
                        or result["negative_false_accept_count"] <= max_negative_false_accepts
                    )
                    and (
                        max_negative_false_accept_rate is None
                        or result["negative_false_accept_rate"] <= max_negative_false_accept_rate
                    )
                    and (max_supported_wrong is None or result["supported_wrong"] <= max_supported_wrong)
                )
                candidates.append(result)

    candidates.sort(
        key=lambda item: (
            bool(item["eligible_for_promotion"]),
            float(item["supported_coverage"]),
            -float(item["negative_false_accept_count"]),
            float(item["supported_precision"]),
            float(item["accuracy"]),
            -float(item["min_similarity"]),
            -float(item["min_margin"]),
            -float(item["min_negative_gap"]),
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    target_policies: dict[str, dict[str, Any]] = {}
    if include_target_policies:
        negative_rows = [row for row in rows if _target_is_negative(row.expected_target, row.expected_behavior)]
        for target in sorted({row.expected_target for row in rows if _target_is_supported_positive(row.expected_target)}):
            target_rows = [row for row in rows if row.expected_target == target]
            target_calibration_rows = target_rows if target_policy_negative_mode == "none" else [*target_rows, *negative_rows]
            target_result = calibrate(
                target_calibration_rows,
                similarity_grid=similarity_grid,
                margin_grid=margin_grid,
                negative_gap_grid=negative_gap_grid,
                min_precision=min_precision if target_min_precision is None else target_min_precision,
                min_coverage=min_coverage,
                require_zero_non_plant_false_accepts=require_zero_non_plant_false_accepts,
                max_negative_false_accepts=max_negative_false_accepts,
                max_negative_false_accept_rate=max_negative_false_accept_rate,
                max_supported_wrong=target_max_supported_wrong,
                include_target_policies=False,
                target_policy_negative_mode=target_policy_negative_mode,
                include_class_policies=False,
                target_class_min_accepted=target_class_min_accepted,
            )
            target_constraints = {
                "min_precision": min_precision if target_min_precision is None else target_min_precision,
                "min_coverage": min_coverage,
                "max_supported_wrong": target_max_supported_wrong,
                "max_cross_part_supported_wrong": target_max_cross_part_supported_wrong,
            }
            target_selected = target_result["selected_policy"]
            target_validation = None
            if target_selected:
                target_validation = evaluate_target_thresholds(
                    rows,
                    target_id=target,
                    min_similarity=float(target_selected["min_similarity"]),
                    min_margin=float(target_selected["min_margin"]),
                    min_negative_gap=float(target_selected.get("min_negative_gap") or 0.0),
                )
                target_selected = {**target_selected, "full_set_validation": target_validation}
                if (
                    target_max_cross_part_supported_wrong is not None
                    and int(target_validation.get("supported_cross_part_wrong") or 0)
                    > int(target_max_cross_part_supported_wrong)
                ):
                    target_selected = None
                elif target_selected and _target_part(target) == "fruit" and int(
                    target_validation.get("supported_cross_part_wrong") or 0
                ) == 0:
                    target_selected = {**target_selected, "allow_part_conflict_override": True}
            best_candidate = target_result["best_candidate"] or {}
            failure_reasons: list[str] = []
            if target_validation:
                best_candidate = {**best_candidate, "full_set_validation": target_validation}
            if not target_selected:
                if float(best_candidate.get("supported_precision") or 0.0) < float(target_constraints["min_precision"]):
                    failure_reasons.append("supported_precision_below_target")
                if float(best_candidate.get("supported_coverage") or 0.0) < float(target_constraints["min_coverage"]):
                    failure_reasons.append("supported_coverage_below_target")
                if (
                    target_constraints["max_supported_wrong"] is not None
                    and int(best_candidate.get("supported_wrong") or 0) > int(target_constraints["max_supported_wrong"])
                ):
                    failure_reasons.append("supported_wrong_above_target")
                cross_part_wrong = int(target_validation.get("supported_cross_part_wrong") or 0) if target_validation else 0
                if (
                    target_constraints["max_cross_part_supported_wrong"] is not None
                    and cross_part_wrong > int(target_constraints["max_cross_part_supported_wrong"])
                ):
                    failure_reasons.append("supported_cross_part_wrong_above_target")
                if (
                    max_negative_false_accepts is not None
                    and int(best_candidate.get("negative_false_accept_count") or 0) > int(max_negative_false_accepts)
                ):
                    failure_reasons.append("negative_false_accepts_above_target")
                if require_zero_non_plant_false_accepts and int(best_candidate.get("non_plant_false_accept_count") or 0):
                    failure_reasons.append("non_plant_false_accepts_present")
            target_policies[target] = {
                "status": "target_specific" if target_selected else "no_eligible_policy",
                "selected_policy": target_selected,
                "best_candidate": best_candidate or target_result["best_candidate"],
                "class_policies": (
                    calibrate_class_policies(
                        rows,
                        target_id=target,
                        similarity_grid=similarity_grid,
                        margin_grid=margin_grid,
                        negative_gap_grid=negative_gap_grid,
                        min_precision=min_precision if target_min_precision is None else target_min_precision,
                        max_supported_wrong=target_max_supported_wrong,
                        max_cross_part_supported_wrong=target_max_cross_part_supported_wrong,
                        min_accepted=target_class_min_accepted,
                        require_zero_non_plant_false_accepts=require_zero_non_plant_false_accepts,
                        max_negative_false_accepts=max_negative_false_accepts,
                        include_negative_rows=True,
                    )
                    if include_class_policies and not target_selected
                    else {}
                ),
                "negative_mode": target_policy_negative_mode,
                "constraints": target_constraints,
                "failure_reasons": failure_reasons,
            }
    return {
        "selected_policy": selected if selected and selected.get("eligible_for_promotion") else None,
        "best_candidate": selected,
        "target_policies": target_policies,
        "candidates": candidates,
    }


def has_runtime_policy(calibration: dict[str, Any]) -> bool:
    if isinstance(calibration.get("selected_policy"), dict):
        return True
    target_policies = calibration.get("target_policies")
    if not isinstance(target_policies, dict):
        return False
    for entry in target_policies.values():
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("selected_policy"), dict):
            return True
        class_policies = entry.get("class_policies")
        if isinstance(class_policies, dict) and any(
            isinstance(class_entry, dict)
            and (
                isinstance(class_entry.get("selected_policy"), dict)
                or isinstance(class_entry.get("exact_class_rescue_policy"), dict)
            )
            for class_entry in class_policies.values()
        ):
            return True
    return False
