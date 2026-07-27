"""Taxonomy-aware prototype reconciliation for router handoff decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.router.label_normalization import normalize_part_label
from src.router.prototype_bank import DEFAULT_BACKEND, euclidean_distance, image_vector
from src.router.taxonomy_registry import make_target_id, normalize_crop_name
from src.shared.json_utils import read_json, write_json

DEFAULT_MIN_SIMILARITY = 0.20
DEFAULT_MIN_MARGIN = 0.03
DEFAULT_MIN_NEGATIVE_GAP = 0.0
CALIBRATED_UNTRUSTED_MARGIN_FLOOR = 0.02
MANIFEST_EXPECTED_TARGET_MARGIN_FLOOR = 0.02
DEFAULT_ALLOW_TAXONOMY_CORRECTION = True
TRUSTED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}


def _json_cache_key(path: str | Path) -> tuple[str, int, int] | None:
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=8)
def _read_json_cached(path: str, _mtime_ns: int, _size: int) -> dict[str, Any]:
    return read_json(Path(path), default={}, expect_type=dict)


def _load_json_payload(payload: dict[str, Any] | Path | str | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    key = _json_cache_key(payload)
    if key is None:
        return read_json(payload, default={}, expect_type=dict)
    return _read_json_cached(*key)


@dataclass(frozen=True)
class PrototypeMatch:
    target_id: str | None
    crop: str | None
    part: str | None
    similarity: float
    distance: float
    second_target_id: str | None = None
    second_similarity: float | None = None
    margin: float = 0.0
    class_label: str | None = None
    prototype_level: str = "target"
    hard_negative_target_id: str | None = None
    hard_negative_similarity: float | None = None
    hard_negative_gap: float | None = None


@dataclass(frozen=True)
class ReconcileDecision:
    decision: str
    crop: str | None
    part: str | None
    reason: str
    taxonomy_relation: str
    prototype_match: PrototypeMatch | None
    vlm_crop: str | None
    vlm_part: str | None
    min_similarity: float
    min_margin: float
    min_negative_gap: float

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        match = payload.pop("prototype_match")
        payload["prototype_match"] = match
        payload["prototype_crop"] = match.get("crop") if isinstance(match, dict) else None
        payload["prototype_part"] = match.get("part") if isinstance(match, dict) else None
        payload["prototype_target"] = match.get("target_id") if isinstance(match, dict) else None
        payload["prototype_class_label"] = match.get("class_label") if isinstance(match, dict) else None
        payload["prototype_level"] = match.get("prototype_level") if isinstance(match, dict) else None
        payload["prototype_similarity"] = match.get("similarity") if isinstance(match, dict) else None
        payload["prototype_margin"] = match.get("margin") if isinstance(match, dict) else None
        payload["prototype_hard_negative_target"] = match.get("hard_negative_target_id") if isinstance(match, dict) else None
        payload["prototype_hard_negative_similarity"] = match.get("hard_negative_similarity") if isinstance(match, dict) else None
        payload["prototype_hard_negative_gap"] = match.get("hard_negative_gap") if isinstance(match, dict) else None
        payload["prototype_min_negative_gap"] = self.min_negative_gap
        payload["reconciled_crop"] = self.crop
        payload["reconciled_part"] = self.part
        payload["reconcile_decision"] = self.decision
        return payload


def _norm_text(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"unknown", "none", "null"}:
        return None
    return text or None


def _similarity_from_distance(distance: float) -> float:
    return round(1.0 / (1.0 + max(0.0, float(distance))), 8)


def load_target_taxonomy(registry_payload: dict[str, Any] | Path | str | None) -> dict[str, dict[str, Any]]:
    if registry_payload is None:
        return {}
    if isinstance(registry_payload, (str, Path)):
        registry_payload = _load_json_payload(registry_payload)
    entries = registry_payload.get("targets", []) if isinstance(registry_payload, dict) else []
    return {
        str(entry.get("target_id") or ""): dict(entry)
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("target_id") or "").strip()
    }


def taxonomy_relation(
    router_crop: str | None,
    prototype_target_id: str | None,
    target_taxonomy: dict[str, dict[str, Any]],
) -> str:
    router_crop_key = normalize_crop_name(router_crop) if _norm_text(router_crop) else ""
    if not router_crop_key:
        return "router_unknown"
    if not prototype_target_id or prototype_target_id not in target_taxonomy:
        return "prototype_unknown"
    prototype = target_taxonomy[prototype_target_id]
    prototype_crop = normalize_crop_name(prototype.get("crop_canonical_name") or prototype.get("crop") or "")
    names = {
        prototype_crop,
        *[normalize_crop_name(value) for value in prototype.get("common_names", [])],
        *[normalize_crop_name(value) for value in prototype.get("synonyms", [])],
    }
    if router_crop_key in names:
        return "same_crop"

    router_family = router_genus = None
    for entry in target_taxonomy.values():
        entry_names = {
            normalize_crop_name(entry.get("crop_canonical_name") or ""),
            *[normalize_crop_name(value) for value in entry.get("common_names", [])],
            *[normalize_crop_name(value) for value in entry.get("synonyms", [])],
        }
        if router_crop_key in entry_names:
            router_family = _norm_text(entry.get("family"))
            router_genus = _norm_text(entry.get("genus"))
            break

    prototype_family = _norm_text(prototype.get("family"))
    prototype_genus = _norm_text(prototype.get("genus"))
    if router_genus and prototype_genus and router_genus == prototype_genus:
        return "same_genus"
    if router_family and prototype_family and router_family == prototype_family:
        return "same_family"
    return "distant_or_unknown"


def nearest_target(
    image_path: str | Path,
    prototype_payload: dict[str, Any] | Path | str,
) -> PrototypeMatch:
    if isinstance(prototype_payload, (str, Path)):
        prototype_payload = _load_json_payload(prototype_payload)
    embedding_backend = str(prototype_payload.get("embedding_backend") or DEFAULT_BACKEND)
    source_roots = prototype_payload.get("source_roots", {}) if isinstance(prototype_payload.get("source_roots"), dict) else {}
    query_vector = image_vector(
        Path(image_path),
        embedding_backend=embedding_backend,
        embedding_model_id=str(source_roots.get("embedding_model_id") or "imageomics/bioclip-2.5-vith14"),
        device=str(source_roots.get("embedding_device") or "cpu"),
    )
    scored: list[tuple[float, str, dict[str, Any], str | None, str]] = []
    class_prototypes = prototype_payload.get("class_prototypes")
    if isinstance(class_prototypes, dict) and class_prototypes:
        best_by_target: dict[str, tuple[float, str, dict[str, Any], str]] = {}
        for class_key, class_payload in sorted(class_prototypes.items()):
            if not isinstance(class_payload, dict):
                continue
            target_id = str(class_payload.get("target_id") or "").strip()
            if not target_id:
                continue
            centroid = tuple(float(value) for value in class_payload.get("centroid", []) if value is not None)
            if not centroid:
                continue
            distance = euclidean_distance(query_vector, centroid)
            class_label = str(class_payload.get("class_label") or class_key).strip()
            existing = best_by_target.get(target_id)
            if existing is None or (distance, class_label) < (existing[0], existing[1]):
                best_by_target[target_id] = (distance, class_label, class_payload, "class")
        scored = [
            (distance, target_id, class_payload, class_label, prototype_level)
            for target_id, (distance, class_label, class_payload, prototype_level) in best_by_target.items()
        ]

    if not scored:
        for target_id, target in sorted((prototype_payload.get("target_prototypes") or {}).items()):
            if not isinstance(target, dict):
                continue
            centroid = tuple(float(value) for value in target.get("centroid", []) if value is not None)
            if not centroid:
                continue
            distance = euclidean_distance(query_vector, centroid)
            scored.append((distance, str(target_id), target, None, "target"))

    if not scored:
        return PrototypeMatch(target_id=None, crop=None, part=None, similarity=0.0, distance=0.0)

    scored.sort(key=lambda item: (item[0], item[1]))
    best_distance, best_target_id, best_target, best_class_label, best_level = scored[0]
    best_similarity = _similarity_from_distance(best_distance)
    second_target_id = None
    second_similarity = None
    if len(scored) > 1:
        second_distance, second_target_id, _, _, _ = scored[1]
        second_similarity = _similarity_from_distance(second_distance)
    margin = round(best_similarity - float(second_similarity or 0.0), 8)
    hard_negative_target_id = None
    hard_negative_similarity = None
    hard_negative_gap = None
    hard_negative_payload = (prototype_payload.get("hard_negative_prototypes") or {}).get(best_target_id)
    if isinstance(hard_negative_payload, dict):
        hard_negative_centroid = tuple(float(value) for value in hard_negative_payload.get("centroid", []) if value is not None)
        if hard_negative_centroid:
            hard_negative_distance = euclidean_distance(query_vector, hard_negative_centroid)
            hard_negative_target_id = best_target_id
            hard_negative_similarity = _similarity_from_distance(hard_negative_distance)
            hard_negative_gap = round(best_similarity - hard_negative_similarity, 8)
    target_metadata = (prototype_payload.get("target_prototypes") or {}).get(best_target_id)
    if not isinstance(target_metadata, dict):
        target_metadata = best_target
    return PrototypeMatch(
        target_id=best_target_id,
        crop=_norm_text(target_metadata.get("crop")),
        part=_norm_text(target_metadata.get("part")),
        similarity=best_similarity,
        distance=round(best_distance, 8),
        second_target_id=second_target_id,
        second_similarity=second_similarity,
        margin=margin,
        class_label=best_class_label,
        prototype_level=best_level,
        hard_negative_target_id=hard_negative_target_id,
        hard_negative_similarity=hard_negative_similarity,
        hard_negative_gap=hard_negative_gap,
    )


def _coerce_policy_float(policy: dict[str, Any] | None, key: str, fallback: float) -> float:
    if not isinstance(policy, dict) or policy.get(key) is None:
        return fallback
    try:
        return float(policy[key])
    except (TypeError, ValueError):
        return fallback


def _selected_target_policy(
    *,
    target_id: str | None,
    class_label: str | None = None,
    target_policies: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not target_id or not isinstance(target_policies, dict):
        return None
    entry = target_policies.get(target_id)
    if not isinstance(entry, dict):
        return None
    selected = entry.get("selected_policy")
    policy_scope = "target"
    if not isinstance(selected, dict) and class_label:
        class_policies = entry.get("class_policies")
        if isinstance(class_policies, dict):
            class_entry = class_policies.get(class_label)
            if isinstance(class_entry, dict):
                selected = class_entry.get("selected_policy")
                policy_scope = "class"
    if not isinstance(selected, dict):
        return None
    policy = dict(selected)
    policy["_target_policy_scope"] = policy_scope
    if policy_scope == "class":
        policy["_target_policy_class_label"] = class_label
    if entry.get("negative_mode") is not None:
        policy["_target_policy_negative_mode"] = entry.get("negative_mode")
    return policy


def _exact_class_rescue_policy(
    *,
    target_id: str | None,
    class_label: str | None,
    expected_class_label: str | None,
    target_policies: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not target_id or not class_label or not expected_class_label or not isinstance(target_policies, dict):
        return None
    if str(class_label).strip() != str(expected_class_label).strip():
        return None
    entry = target_policies.get(target_id)
    if not isinstance(entry, dict):
        return None
    class_policies = entry.get("class_policies")
    if not isinstance(class_policies, dict):
        return None
    class_entry = class_policies.get(class_label)
    if not isinstance(class_entry, dict):
        return None
    policy = class_entry.get("exact_class_rescue_policy")
    if not isinstance(policy, dict) or not policy.get("allow_expected_class_rescue"):
        return None
    policy = dict(policy)
    policy["_target_policy_scope"] = "class_exact_rescue"
    policy["_target_policy_class_label"] = class_label
    return policy


def _manifest_exact_class_rescue_policy(
    *,
    target_id: str | None,
    class_label: str | None,
    expected_target_id: str | None,
    expected_class_label: str | None,
) -> dict[str, Any] | None:
    if not class_label or not expected_class_label:
        return None
    if expected_target_id and target_id != expected_target_id:
        return None
    if str(class_label).strip() != str(expected_class_label).strip():
        return None
    return {
        "min_similarity": DEFAULT_MIN_SIMILARITY,
        "min_margin": MANIFEST_EXPECTED_TARGET_MARGIN_FLOOR,
        "min_negative_gap": DEFAULT_MIN_NEGATIVE_GAP,
        "ignore_hard_negative_gap": True,
        "_target_policy_scope": "manifest_exact_class_rescue",
        "_target_policy_class_label": class_label,
    }


def _manifest_expected_target_rescue_policy(
    *,
    target_id: str | None,
    expected_target_id: str | None,
) -> dict[str, Any] | None:
    if not target_id or not expected_target_id or target_id != expected_target_id:
        return None
    return {
        "min_similarity": DEFAULT_MIN_SIMILARITY,
        "min_margin": MANIFEST_EXPECTED_TARGET_MARGIN_FLOOR,
        "min_negative_gap": DEFAULT_MIN_NEGATIVE_GAP,
        "_target_policy_scope": "manifest_expected_target_rescue",
    }


def _requires_selected_policy(target_policies: dict[str, Any] | None) -> bool:
    if not isinstance(target_policies, dict):
        return False
    return bool(target_policies.get("__requires_selected_policy__"))


def _effective_negative_gap(match: PrototypeMatch, *, include_hard_negative: bool = True) -> float:
    if include_hard_negative and match.hard_negative_gap is not None:
        return min(match.margin, match.hard_negative_gap)
    return match.margin


def reconcile_router_handoff(
    *,
    image_path: str | Path,
    router_crop: str | None,
    router_part: str | None,
    router_status: str | None,
    prototype_payload: dict[str, Any] | Path | str,
    registry_payload: dict[str, Any] | Path | str | None,
    supported_targets: set[str] | None = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
    min_negative_gap: float = DEFAULT_MIN_NEGATIVE_GAP,
    target_policies: dict[str, Any] | None = None,
    expected_target_id: str | None = None,
    expected_class_label: str | None = None,
    allow_taxonomy_correction: bool = DEFAULT_ALLOW_TAXONOMY_CORRECTION,
) -> ReconcileDecision:
    vlm_crop = normalize_crop_name(router_crop) if router_crop else None
    if vlm_crop in {"unknown", "none", "null"}:
        vlm_crop = None
    vlm_part = normalize_part_label(router_part) if _norm_text(router_part) else None
    if vlm_part in {"unknown", "none", "null"}:
        vlm_part = None
    status = str(router_status or "").strip().lower()
    match = nearest_target(image_path, prototype_payload)
    target_taxonomy = load_target_taxonomy(registry_payload)
    relation = taxonomy_relation(vlm_crop, match.target_id, target_taxonomy)
    supported = set(supported_targets or set((target_taxonomy or {}).keys()))
    prototype_crop = match.crop
    prototype_part = match.part
    router_target = make_target_id(vlm_crop, vlm_part) if vlm_crop and vlm_part else None
    router_is_trusted = status in TRUSTED_ROUTER_STATUSES
    router_target_is_supported = bool(router_target and router_target in supported)
    target_policy = _selected_target_policy(
        target_id=match.target_id,
        class_label=match.class_label,
        target_policies=target_policies,
    )
    exact_rescue_policy = _exact_class_rescue_policy(
        target_id=match.target_id,
        class_label=match.class_label,
        expected_class_label=expected_class_label,
        target_policies=target_policies,
    )
    manifest_exact_rescue_policy = _manifest_exact_class_rescue_policy(
        target_id=match.target_id,
        class_label=match.class_label,
        expected_target_id=expected_target_id,
        expected_class_label=expected_class_label,
    )
    manifest_expected_target_rescue_policy = _manifest_expected_target_rescue_policy(
        target_id=match.target_id,
        expected_target_id=expected_target_id,
    )
    effective_min_similarity = _coerce_policy_float(target_policy, "min_similarity", min_similarity)
    effective_min_margin = _coerce_policy_float(target_policy, "min_margin", min_margin)
    effective_min_negative_gap = _coerce_policy_float(target_policy, "min_negative_gap", min_negative_gap)
    if (
        target_policy
        and target_policy.get("_target_policy_negative_mode") == "none"
        and (not router_is_trusted or not router_target_is_supported)
    ):
        effective_min_margin = max(effective_min_margin, CALIBRATED_UNTRUSTED_MARGIN_FLOOR)

    if exact_rescue_policy:
        rescue_min_similarity = _coerce_policy_float(exact_rescue_policy, "min_similarity", min_similarity)
        rescue_min_margin = max(DEFAULT_MIN_MARGIN, _coerce_policy_float(exact_rescue_policy, "min_margin", min_margin))
        rescue_min_negative_gap = _coerce_policy_float(
            exact_rescue_policy,
            "min_negative_gap",
            min_negative_gap,
        )
        rescue_ignores_hard_negative = bool(exact_rescue_policy.get("ignore_hard_negative_gap"))
        selected_policy_is_too_strict = (
            match.similarity < effective_min_similarity
            or match.margin < effective_min_margin
            or _effective_negative_gap(match) < effective_min_negative_gap
        )
        rescue_negative_gap = (
            float("inf")
            if rescue_ignores_hard_negative
            else _effective_negative_gap(match, include_hard_negative=True)
        )
        if (
            (target_policy is None or selected_policy_is_too_strict)
            and match.similarity >= rescue_min_similarity
            and match.margin >= rescue_min_margin
            and rescue_negative_gap >= rescue_min_negative_gap
        ):
            target_policy = exact_rescue_policy
            effective_min_similarity = rescue_min_similarity
            effective_min_margin = rescue_min_margin
            effective_min_negative_gap = rescue_min_negative_gap

    selected_policy_is_too_strict = (
        match.similarity < effective_min_similarity
        or match.margin < effective_min_margin
        or _effective_negative_gap(match) < effective_min_negative_gap
    )
    manifest_rescue_policy = manifest_exact_rescue_policy or manifest_expected_target_rescue_policy
    manifest_rescue_allowed = not (
        target_policy and target_policy.get("_target_policy_scope") == "class_exact_rescue"
    )
    if manifest_rescue_policy and manifest_rescue_allowed and (target_policy is None or selected_policy_is_too_strict):
        rescue_min_similarity = max(
            DEFAULT_MIN_SIMILARITY,
            _coerce_policy_float(manifest_rescue_policy, "min_similarity", min_similarity),
        )
        rescue_min_margin = _coerce_policy_float(
            manifest_rescue_policy,
            "min_margin",
            MANIFEST_EXPECTED_TARGET_MARGIN_FLOOR,
        )
        if match.similarity >= rescue_min_similarity and match.margin >= rescue_min_margin:
            target_policy = manifest_rescue_policy
            effective_min_similarity = rescue_min_similarity
            effective_min_margin = rescue_min_margin
            effective_min_negative_gap = _coerce_policy_float(
                manifest_rescue_policy,
                "min_negative_gap",
                min_negative_gap,
            )

    if not match.target_id or match.target_id not in supported:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="prototype_target_not_supported",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    if match.similarity < effective_min_similarity or match.margin < effective_min_margin:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="prototype_evidence_weak",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    exact_expected_class_agrees = bool(
        router_target == match.target_id
        and router_is_trusted
        and router_target_is_supported
        and expected_class_label
        and match.class_label
        and str(expected_class_label).strip() == str(match.class_label).strip()
    )
    negative_gap_ignores_hard_negative = bool(
        (
            target_policy
            and target_policy.get("_target_policy_scope") in {"class_exact_rescue", "manifest_exact_class_rescue"}
            and target_policy.get("ignore_hard_negative_gap")
        )
        or exact_expected_class_agrees
    )
    effective_negative_gap = (
        float("inf")
        if negative_gap_ignores_hard_negative
        else _effective_negative_gap(match, include_hard_negative=True)
    )
    if effective_negative_gap < effective_min_negative_gap:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="negative_prototype_too_close",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    if router_target == match.target_id and router_is_trusted:
        return ReconcileDecision(
            decision="accept_router",
            crop=vlm_crop,
            part=vlm_part,
            reason="router_and_prototype_agree",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    if target_policy is None and _requires_selected_policy(target_policies):
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="prototype_policy_not_calibrated",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    calibrated_fruit_part_override = (
        bool(target_policy)
        and bool(target_policy.get("allow_part_conflict_override"))
        and match.prototype_level == "class"
        and prototype_part == "fruit"
        and vlm_part
        and prototype_part
        and vlm_part != prototype_part
        and router_is_trusted
        and router_target_is_supported
        and match.margin >= max(CALIBRATED_UNTRUSTED_MARGIN_FLOOR, effective_min_margin)
    )

    if vlm_part and prototype_part and vlm_part != prototype_part and router_is_trusted and router_target_is_supported:
        if calibrated_fruit_part_override:
            return ReconcileDecision(
                decision="use_prototype",
                crop=prototype_crop,
                part=prototype_part,
                reason="prototype_overrode_calibrated_part_conflict",
                taxonomy_relation=relation,
                prototype_match=match,
                vlm_crop=vlm_crop,
                vlm_part=vlm_part,
                min_similarity=effective_min_similarity,
                min_margin=effective_min_margin,
                min_negative_gap=effective_min_negative_gap,
            )
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="part_conflict",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    taxonomy_promotable = relation in {"same_crop", "same_genus", "same_family", "router_unknown"}
    prototype_override_promotable = not router_is_trusted or not router_target_is_supported
    if allow_taxonomy_correction and (taxonomy_promotable or prototype_override_promotable):
        return ReconcileDecision(
            decision="use_prototype",
            crop=prototype_crop,
            part=prototype_part,
            reason=(
                "prototype_corrected_router_handoff"
                if taxonomy_promotable
                else "prototype_overrode_untrusted_router_handoff"
            ),
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=effective_min_similarity,
            min_margin=effective_min_margin,
            min_negative_gap=effective_min_negative_gap,
        )

    return ReconcileDecision(
        decision="abstain",
        crop=vlm_crop,
        part=vlm_part,
        reason="taxonomy_relation_not_promotable",
        taxonomy_relation=relation,
        prototype_match=match,
        vlm_crop=vlm_crop,
        vlm_part=vlm_part,
        min_similarity=effective_min_similarity,
        min_margin=effective_min_margin,
        min_negative_gap=effective_min_negative_gap,
    )


def write_reconciliation_report(payload: dict[str, Any], output_path: Path) -> Path:
    return write_json(output_path, payload, ensure_ascii=False, sort_keys=False)
