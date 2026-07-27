"""Validation helpers for continual training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import torch

from src.ood.sure_scoring import compute_ds_f1
from src.training.services.metrics import compute_ood_detection_metrics
from src.training.types import EvaluationArtifactsPayload, ValidationReport


@dataclass
class _ArtifactMetricState:
    conformal_hits: int = 0
    conformal_total: int = 0
    conformal_set_size_total: int = 0
    mixed_ood_labels: List[int] = field(default_factory=list)
    mixed_ood_scores: List[float] = field(default_factory=list)
    mixed_ood_scores_by_method: Dict[str, List[float]] = field(default_factory=dict)
    mixed_ood_sample_types: List[Optional[str]] = field(default_factory=list)
    primary_score_method: str = "ensemble"
    semantic_labels: List[int] = field(default_factory=list)
    semantic_preds: List[int] = field(default_factory=list)
    confidence_labels: List[int] = field(default_factory=list)
    confidence_preds: List[int] = field(default_factory=list)


def _safe_label_name(idx_to_class: Dict[int, str], label: int) -> str:
    if int(label) < 0:
        return ""
    return str(idx_to_class.get(int(label), int(label)))


def _resolve_candidate_score_payload(detector: Any, ood: Dict[str, Any]) -> tuple[str, Dict[str, torch.Tensor]]:
    primary_method = str(
        ood.get("primary_score_method", getattr(detector, "primary_score_method", "ensemble")) or "ensemble"
    ).strip().lower()
    candidate_scores = ood.get("candidate_scores")
    if not isinstance(candidate_scores, dict) or not candidate_scores:
        return primary_method, {}
    return primary_method, {
        str(name): value
        for name, value in candidate_scores.items()
        if torch.is_tensor(value)
    }

def _build_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if labels.numel() <= 0 or preds.numel() <= 0:
        return confusion
    valid_mask = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    if not bool(valid_mask.any().item()):
        return confusion
    valid_labels = labels[valid_mask]
    valid_preds = preds[valid_mask]
    flat_index = (valid_labels * num_classes + valid_preds).to(torch.long)
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    confusion = counts.reshape(num_classes, num_classes)
    return confusion


def _update_detector_artifact_state_for_batch(
    state: _ArtifactMetricState,
    *,
    detector: Any,
    idx_to_class: Dict[int, str],
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    ood_loader_present: bool,
    is_ood_loader: bool,
    batch_sample_types: Optional[List[Optional[str]]] = None,
    ood_payload: Optional[Dict[str, Any]] = None,
) -> None:
    ood = dict(ood_payload or detector.score(features, logits, predicted_labels=predictions))
    primary_method, candidate_scores = _resolve_candidate_score_payload(detector, ood)
    primary_scores = ood.get("primary_score")
    if not torch.is_tensor(primary_scores):
        primary_scores = candidate_scores.get(primary_method)
        if not torch.is_tensor(primary_scores):
            primary_scores = candidate_scores.get("ensemble")

    if ood_loader_present:
        batch_ood_labels = [1 if is_ood_loader else 0] * int(labels.shape[0])
        state.mixed_ood_labels.extend(batch_ood_labels)
        if batch_sample_types is not None and len(batch_sample_types) == int(labels.shape[0]):
            state.mixed_ood_sample_types.extend(list(batch_sample_types))
        else:
            state.mixed_ood_sample_types.extend([None] * int(labels.shape[0]))
        state.primary_score_method = primary_method
        if torch.is_tensor(primary_scores):
            state.mixed_ood_scores.extend(float(score) for score in primary_scores.detach().cpu().tolist())
        for method_name, method_scores in candidate_scores.items():
            bucket = state.mixed_ood_scores_by_method.setdefault(str(method_name), [])
            bucket.extend(float(score) for score in method_scores.detach().cpu().tolist())

    if getattr(detector, "sure_enabled", False) and ood_loader_present and "sure_semantic_ood" in ood:
        state.semantic_labels.extend([1 if is_ood_loader else 0] * int(labels.shape[0]))
        state.semantic_preds.extend(int(flag) for flag in ood["sure_semantic_ood"].detach().cpu().tolist())
        if is_ood_loader:
            state.confidence_labels.extend([1] * int(labels.shape[0]))
        else:
            state.confidence_labels.extend(int(flag) for flag in (predictions != labels).detach().cpu().tolist())
        state.confidence_preds.extend(int(flag) for flag in ood["sure_confidence_reject"].detach().cpu().tolist())

    if (
        not is_ood_loader
        and getattr(detector, "conformal_enabled", False)
        and getattr(detector, "conformal_qhat", None) is not None
    ):
        for feat, logit, label in zip(features, logits, labels):
            pred_set = detector.build_conformal_set(feat, logit, idx_to_class)
            true_label = idx_to_class.get(int(label.item()), str(int(label.item())))
            state.conformal_hits += int(true_label in pred_set)
            state.conformal_total += 1
            state.conformal_set_size_total += len(pred_set)


def _infer_ood_type_from_path(image_path: Any, *, split_name: str) -> str:
    try:
        parts = list(getattr(image_path, "parts", []))
    except (TypeError, AttributeError):
        parts = []
    if not parts:
        text = str(image_path)
        parts = [part for part in text.replace("\\", "/").split("/") if part]
    split_index = -1
    for index, part in enumerate(parts):
        if str(part).lower() == str(split_name).lower():
            split_index = index
            break
    if split_index < 0:
        return "unlabeled"
    relative_parts = parts[split_index + 1 :]
    if len(relative_parts) <= 1:
        return "unlabeled"
    return str(relative_parts[0])


def _resolve_loader_image_paths(eval_loader: Iterable[Dict[str, torch.Tensor]]) -> Optional[List[str]]:
    dataset = getattr(eval_loader, "dataset", None)
    image_paths = getattr(dataset, "image_paths", None)
    if not isinstance(image_paths, list):
        return None
    return [path.as_posix() if hasattr(path, "as_posix") else str(path).replace("\\", "/") for path in image_paths]


def _resolve_loader_sample_types(
    eval_loader: Iterable[Dict[str, torch.Tensor]], *, is_ood_loader: bool
) -> Optional[List[str]]:
    if not is_ood_loader:
        return None
    dataset = getattr(eval_loader, "dataset", None)
    sample_types = getattr(dataset, "ood_sample_types", None)
    if isinstance(sample_types, list) and sample_types:
        return [str(sample_type or "unlabeled") for sample_type in sample_types]
    image_paths = getattr(dataset, "image_paths", None)
    split_name = str(getattr(dataset, "split", "") or "")
    if not image_paths or split_name.lower() != "ood":
        return None
    return [_infer_ood_type_from_path(path, split_name=split_name) for path in list(image_paths)]


def _build_ood_type_breakdown(state: _ArtifactMetricState) -> Dict[str, Any]:
    if not state.mixed_ood_sample_types or not state.mixed_ood_scores_by_method:
        return {}

    id_indices = [idx for idx, label in enumerate(state.mixed_ood_labels) if int(label) == 0]
    ood_types = sorted(
        {
            str(sample_type)
            for sample_type, label in zip(state.mixed_ood_sample_types, state.mixed_ood_labels)
            if sample_type and int(label) == 1
        }
    )
    if not id_indices or not ood_types:
        return {}

    breakdown: Dict[str, Any] = {}
    for ood_type in ood_types:
        ood_indices = [
            idx
            for idx, (sample_type, label) in enumerate(zip(state.mixed_ood_sample_types, state.mixed_ood_labels))
            if int(label) == 1 and sample_type == ood_type
        ]
        if not ood_indices:
            continue
        method_metrics: Dict[str, Any] = {}
        for method_name, all_scores in state.mixed_ood_scores_by_method.items():
            labels = ([0] * len(id_indices)) + ([1] * len(ood_indices))
            scores = [float(all_scores[idx]) for idx in id_indices] + [float(all_scores[idx]) for idx in ood_indices]
            metrics = compute_ood_detection_metrics(ood_labels=labels, ood_scores=scores)
            ood_samples_val = metrics.get("ood_samples")
            in_dist_samples_val = metrics.get("in_distribution_samples")
            method_metrics[str(method_name)] = {
                "ood_auroc": metrics["ood_auroc"],
                "ood_false_positive_rate": metrics["ood_false_positive_rate"],
                "ood_samples": None if ood_samples_val is None else int(ood_samples_val),
                "in_distribution_samples": None if in_dist_samples_val is None else int(in_dist_samples_val),
            }
        breakdown[str(ood_type)] = {
            "ood_type": str(ood_type),
            "sample_count": int(len(ood_indices)),
            "primary_score_method": str(state.primary_score_method or "ensemble"),
            "metrics": dict(method_metrics.get(str(state.primary_score_method or "ensemble"), {})),
            "method_metrics": method_metrics,
        }
    return breakdown


def _append_prediction_rows_for_batch(
    prediction_rows: List[Dict[str, Any]],
    *,
    idx_to_class: Dict[int, str],
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    image_paths: Optional[List[str]],
    image_offset: int,
    sample_origin: str,
    split_name: str,
    batch_sample_types: Optional[List[Optional[str]]] = None,
    ood_payload: Optional[Dict[str, Any]] = None,
) -> None:
    probabilities = torch.softmax(logits.detach(), dim=1)
    class_confidence = probabilities.max(dim=1).values.detach().cpu().tolist()
    label_values = labels.detach().cpu().tolist()
    prediction_values = predictions.detach().cpu().tolist()

    primary_method = ""
    primary_scores: List[Any] = []
    ood_flags: List[Any] = []
    if isinstance(ood_payload, dict) and ood_payload:
        primary_method = str(ood_payload.get("primary_score_method", "") or "").strip()
        primary_score_value = ood_payload.get("primary_score")
        if torch.is_tensor(primary_score_value):
            primary_scores = primary_score_value.detach().cpu().tolist()
        predicted_ood_value = ood_payload.get("is_ood")
        if torch.is_tensor(predicted_ood_value):
            ood_flags = predicted_ood_value.detach().cpu().tolist()

    for index, (label, prediction) in enumerate(zip(label_values, prediction_values)):
        image_path = ""
        absolute_index = image_offset + index
        if image_paths is not None and absolute_index < len(image_paths):
            image_path = str(image_paths[absolute_index])

        ood_type = ""
        if batch_sample_types is not None and index < len(batch_sample_types):
            ood_type = str(batch_sample_types[index] or "")

        prediction_rows.append(
            {
                "sample_origin": str(sample_origin),
                "split_name": str(split_name),
                "image_path": image_path,
                "ood_type": ood_type,
                "true_index": (None if int(label) < 0 else int(label)),
                "true_label": _safe_label_name(idx_to_class, int(label)),
                "pred_index": int(prediction),
                "pred_label": _safe_label_name(idx_to_class, int(prediction)),
                "is_correct": (None if int(label) < 0 else bool(int(prediction) == int(label))),
                "class_confidence": float(class_confidence[index]),
                "ood_primary_score_method": str(primary_method),
                "ood_primary_score": (
                    None
                    if not primary_scores or index >= len(primary_scores)
                    else float(primary_scores[index])
                ),
                "ood_predicted": (
                    None
                    if not ood_flags or index >= len(ood_flags)
                    else bool(ood_flags[index])
                ),
            }
        )


def _compute_classification_metrics(
    confusion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    tp = confusion.diag().to(torch.float32)
    support = confusion.sum(dim=1).to(torch.float32)
    predicted_support = confusion.sum(dim=0).to(torch.float32)

    precision = torch.zeros_like(tp)
    recall = torch.zeros_like(tp)
    precision_mask = predicted_support > 0
    recall_mask = support > 0
    precision[precision_mask] = tp[precision_mask] / predicted_support[precision_mask]
    recall[recall_mask] = tp[recall_mask] / support[recall_mask]

    f1 = torch.zeros_like(tp)
    denom = precision + recall
    valid_f1 = denom > 0
    f1[valid_f1] = (2.0 * precision[valid_f1] * recall[valid_f1]) / denom[valid_f1]

    supported_class_mask = support > 0
    if bool(supported_class_mask.any().item()):
        macro_precision = float(precision[supported_class_mask].mean().item())
        macro_recall = float(recall[supported_class_mask].mean().item())
        macro_f1 = float(f1[supported_class_mask].mean().item())
        balanced_accuracy = macro_recall
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        balanced_accuracy = 0.0

    return tp, support, precision, recall, f1, macro_precision, macro_recall, macro_f1, balanced_accuracy


def _build_per_class_metrics(
    trainer: Any,
    *,
    num_classes: int,
    recall: torch.Tensor,
    recall_mask: torch.Tensor,
    support: torch.Tensor,
) -> tuple[Dict[str, float], Dict[str, int]]:
    idx_to_class = {idx: name for name, idx in trainer.class_to_idx.items()}
    per_class_accuracy: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}
    for class_index in range(num_classes):
        class_name = str(idx_to_class.get(class_index, class_index))
        per_class_accuracy[class_name] = (
            float(recall[class_index].item()) if bool(recall_mask[class_index].item()) else 0.0
        )
        per_class_support[class_name] = int(support[class_index].item())
    return per_class_accuracy, per_class_support


def _rank_worst_classes(
    *,
    num_classes: int,
    class_to_idx: Dict[str, int],
    per_class_accuracy: Dict[str, float],
    per_class_support: Dict[str, int],
) -> List[Dict[str, Any]]:
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    def _worst_class_sort_key(item: Dict[str, Any]) -> tuple[float, int, str]:
        return (float(item["accuracy"]), -int(item["support"]), str(item["class_name"]))

    return sorted(
        [
            {
                "class_name": class_name,
                "class_index": int(class_index),
                "accuracy": float(per_class_accuracy[class_name]),
                "support": int(per_class_support[class_name]),
            }
            for class_index, class_name in [(idx, str(idx_to_class.get(idx, idx))) for idx in range(num_classes)]
            if per_class_support[class_name] > 0
        ],
        key=_worst_class_sort_key,
    )


def _can_collect_detector_artifacts(trainer: Any, detector: Any) -> bool:
    if detector is None or not hasattr(detector, "calibration_issue"):
        return False
    if detector.calibration_issue() is not None:
        return False
    idx_to_class = {idx: name for name, idx in dict(getattr(trainer, "class_to_idx", {})).items()}
    return bool(idx_to_class)


def _update_detector_artifact_state(
    state: _ArtifactMetricState,
    *,
    detector: Any,
    idx_to_class: Dict[int, str],
    trainer: Any,
    eval_loader: Iterable[Dict[str, torch.Tensor]],
    ood_loader_present: bool,
    is_ood_loader: bool,
    prediction_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    sample_types = _resolve_loader_sample_types(eval_loader, is_ood_loader=is_ood_loader)
    image_paths = _resolve_loader_image_paths(eval_loader)
    sample_offset = 0
    trainer.set_eval_mode()
    with torch.inference_mode():
        for batch in eval_loader:
            images = batch["images"].to(trainer.device, non_blocking=True)
            labels = batch["labels"].to(trainer.device, non_blocking=True)
            features = trainer.encode(images)
            score_features = (
                trainer.prepare_features_for_scoring(features)
                if hasattr(trainer, "prepare_features_for_scoring")
                else features
            )
            logits = trainer.classifier(score_features)
            predictions = torch.argmax(logits, dim=1)
            ood_payload = detector.score(score_features, logits, predicted_labels=predictions)
            batch_size = int(labels.shape[0])
            batch_sample_types = None
            if sample_types is not None:
                batch_sample_types = [str(value) for value in sample_types[sample_offset : sample_offset + batch_size]]
            _update_detector_artifact_state_for_batch(
                state,
                detector=detector,
                idx_to_class=idx_to_class,
                features=score_features,
                logits=logits,
                labels=labels,
                predictions=predictions,
                ood_loader_present=ood_loader_present,
                is_ood_loader=is_ood_loader,
                batch_sample_types=cast(List[Optional[str]], batch_sample_types),
                ood_payload=ood_payload,
            )
            if prediction_rows is not None:
                _append_prediction_rows_for_batch(
                    prediction_rows,
                    idx_to_class=idx_to_class,
                    logits=logits,
                    labels=labels,
                    predictions=predictions,
                    image_paths=image_paths,
                    image_offset=sample_offset,
                    sample_origin=("ood" if is_ood_loader else "in_distribution"),
                    split_name=str(getattr(getattr(eval_loader, "dataset", None), "split", "") or ""),
                    batch_sample_types=cast(List[Optional[str]], batch_sample_types),
                    ood_payload=ood_payload,
                )
            sample_offset += batch_size


def _finalize_artifact_metric_state(payload: EvaluationArtifactsPayload, state: _ArtifactMetricState) -> None:
    if state.conformal_total > 0:
        payload.conformal_empirical_coverage = float(state.conformal_hits) / float(state.conformal_total)
        payload.conformal_avg_set_size = float(state.conformal_set_size_total) / float(state.conformal_total)
        payload.context["conformal_eval_samples"] = int(state.conformal_total)

    if state.mixed_ood_labels and 0 in state.mixed_ood_labels and 1 in state.mixed_ood_labels:
        payload.ood_labels = list(state.mixed_ood_labels)
        payload.ood_scores = list(state.mixed_ood_scores)
        payload.ood_primary_score_method = str(state.primary_score_method or "ensemble")
        payload.ood_scores_by_method = {
            str(method_name): list(values)
            for method_name, values in state.mixed_ood_scores_by_method.items()
        }
        payload.ood_type_breakdown = _build_ood_type_breakdown(state)
        payload.context["ood_eval_in_distribution_samples"] = int(
            sum(1 for item in state.mixed_ood_labels if item == 0)
        )
        payload.context["ood_eval_ood_samples"] = int(
            sum(1 for item in state.mixed_ood_labels if item == 1)
        )
        payload.context["ood_primary_score_method"] = payload.ood_primary_score_method
        payload.context["ood_score_methods"] = sorted(payload.ood_scores_by_method.keys())
        if payload.ood_type_breakdown:
            payload.context["ood_types"] = sorted(payload.ood_type_breakdown.keys())

    if state.semantic_labels and 0 in state.semantic_labels and 1 in state.semantic_labels and state.confidence_labels:
        ds_f1 = compute_ds_f1(
            torch.tensor(state.semantic_labels, dtype=torch.long),
            torch.tensor(state.confidence_labels, dtype=torch.long),
            torch.tensor(state.semantic_preds, dtype=torch.long),
            torch.tensor(state.confidence_preds, dtype=torch.long),
        )
        payload.sure_ds_f1 = float(ds_f1["ds_f1"])
        payload.context["sure_semantic_f1"] = float(ds_f1["semantic_f1"])
        payload.context["sure_confidence_f1"] = float(ds_f1["confidence_f1"])


def _evaluate_model_core(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    detector: Any = None,
    artifact_state: Optional[_ArtifactMetricState] = None,
    idx_to_class: Optional[Dict[int, str]] = None,
    ood_loader_present: bool = False,
    prediction_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[ValidationReport, List[int], List[int]]]:
    if getattr(trainer, "classifier", None) is None:
        return None

    trainer.set_eval_mode()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    image_paths = _resolve_loader_image_paths(loader)
    image_offset = 0
    split_name = str(getattr(getattr(loader, "dataset", None), "split", "") or "")

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(trainer.device, non_blocking=True)
            labels = batch["labels"].to(trainer.device, non_blocking=True)

            if artifact_state is not None and detector is not None:
                features = trainer.encode(images)
                score_features = (
                    trainer.prepare_features_for_scoring(features)
                    if hasattr(trainer, "prepare_features_for_scoring")
                    else features
                )
                logits = trainer.classifier(score_features)
            else:
                features = None
                score_features = None
                try:
                    logits = trainer.forward_logits(images, apply_scoring_adjustments=True)
                except TypeError:
                    logits = trainer.forward_logits(images)

            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.0)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * float(batch_size)
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            all_labels.append(labels.detach().cpu())
            all_preds.append(predictions.detach().cpu())

            ood_payload = None
            if artifact_state is not None and detector is not None and features is not None:
                ood_payload = detector.score(score_features, logits, predicted_labels=predictions)
                _update_detector_artifact_state_for_batch(
                    artifact_state,
                    detector=detector,
                    idx_to_class=idx_to_class or {},
                    features=score_features,
                    logits=logits,
                    labels=labels,
                    predictions=predictions,
                    ood_loader_present=ood_loader_present,
                    is_ood_loader=False,
                    ood_payload=ood_payload,
                )
            if prediction_rows is not None:
                _append_prediction_rows_for_batch(
                    prediction_rows,
                    idx_to_class=idx_to_class or {},
                    logits=logits,
                    labels=labels,
                    predictions=predictions,
                    image_paths=image_paths,
                    image_offset=image_offset,
                    sample_origin="in_distribution",
                    split_name=split_name,
                    ood_payload=ood_payload,
                )
            image_offset += batch_size

    if total_samples <= 0:
        return None

    val_loss = float(total_loss / max(1, total_samples))
    val_accuracy = float(total_correct / max(1, total_samples))

    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)

    num_classes = max(1, int(getattr(trainer.classifier, "out_features", 1)))
    confusion = (
        _build_confusion_matrix(labels, preds, num_classes)
        if labels.numel() > 0
        else torch.zeros((num_classes, num_classes), dtype=torch.long)
    )
    (
        _tp,
        support,
        _precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        macro_f1,
        balanced_accuracy,
    ) = _compute_classification_metrics(confusion)
    recall_mask = support > 0

    weighted_f1 = float((f1 * support).sum().item() / max(1.0, float(support.sum().item())))

    per_class_accuracy, per_class_support = _build_per_class_metrics(
        trainer,
        num_classes=num_classes,
        recall=recall,
        recall_mask=recall_mask,
        support=support,
    )
    ranked_worst = _rank_worst_classes(
        num_classes=num_classes,
        class_to_idx=trainer.class_to_idx,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
    )

    report = ValidationReport(
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        balanced_accuracy=balanced_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        worst_classes=ranked_worst[:3],
    )
    return report, labels.tolist(), preds.tolist()


def evaluate_model(trainer: Any, loader: Iterable[Dict[str, torch.Tensor]]) -> Optional[ValidationReport]:
    result = _evaluate_model_core(trainer, loader)
    return None if result is None else result[0]


def evaluate_model_with_predictions(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Optional[Tuple[ValidationReport, List[int], List[int]]]:
    return _evaluate_model_core(trainer, loader)


def evaluate_model_with_artifact_metrics(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    ood_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
) -> Optional[EvaluationArtifactsPayload]:
    detector = getattr(trainer, "ood_detector", None)
    idx_to_class = {idx: name for name, idx in dict(getattr(trainer, "class_to_idx", {})).items()}
    can_collect = _can_collect_detector_artifacts(trainer, detector)
    state = _ArtifactMetricState() if can_collect else None
    prediction_rows: List[Dict[str, Any]] = []

    result = _evaluate_model_core(
        trainer,
        loader,
        detector=detector if can_collect else None,
        artifact_state=state,
        idx_to_class=idx_to_class,
        ood_loader_present=ood_loader is not None,
        prediction_rows=prediction_rows,
    )
    if result is None:
        return None

    report, y_true, y_pred = result
    payload = EvaluationArtifactsPayload(
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        prediction_rows=prediction_rows,
    )
    if not can_collect or state is None:
        return payload

    if ood_loader is not None:
        _update_detector_artifact_state(
            state,
            detector=detector,
            idx_to_class=idx_to_class,
            trainer=trainer,
            eval_loader=ood_loader,
            ood_loader_present=True,
            is_ood_loader=True,
            prediction_rows=prediction_rows,
        )
    _finalize_artifact_metric_state(payload, state)

    return payload
