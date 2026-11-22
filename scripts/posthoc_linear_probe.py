"""Post-hoc baseline: a linear probe on frozen DINOv3 features, scored next to a released adapter.

This is an analysis added after the project was frozen. It trains nothing that ships: the probe is fitted
in memory on the target's existing train split, and the released adapter is only read. Both are scored on
the same in-distribution test split and the same held-out OOD split, so the numbers are directly comparable.

Reported per model:
- accuracy, balanced accuracy and macro F1 on the ID test split;
- OOD AUROC (OOD = positive) for MSP, max logit, energy and kNN scores (probe), and the adapter's own
  calibrated primary score;
- selective classification on the ID test split: AURC and AUGRC (Traub et al., 2024) with the model's
  softmax confidence as the acceptance score.

Per-sample rows are written next to the summary so the curves can be redrawn without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
KNN_K = 50


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    return features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)


def knn_distance(query: np.ndarray, bank: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Distance to the k-th nearest normalized training feature (Sun et al., 2022); larger = more OOD."""
    query_n = _l2_normalize(np.asarray(query, dtype=np.float64))
    bank_n = _l2_normalize(np.asarray(bank, dtype=np.float64))
    k = max(1, min(int(k), bank_n.shape[0]))
    distances = np.sqrt(np.clip(2.0 - 2.0 * query_n @ bank_n.T, 0.0, None))
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def ood_scores_from_logits(logits: np.ndarray) -> Dict[str, np.ndarray]:
    """Scores where larger means more likely OOD."""
    logits = np.asarray(logits, dtype=np.float64)
    max_logit = logits.max(axis=1)
    energy = max_logit + np.log(np.exp(logits - max_logit[:, None]).sum(axis=1))
    return {
        "msp": -softmax(logits).max(axis=1),
        "max_logit": -max_logit,
        "energy": -energy,
    }


def ood_auroc(id_scores: Sequence[float], ood_scores: Sequence[float]) -> Optional[float]:
    """AUROC with OOD as the positive class; ties count half."""
    id_arr = np.asarray(id_scores, dtype=np.float64)
    ood_arr = np.asarray(ood_scores, dtype=np.float64)
    if id_arr.size == 0 or ood_arr.size == 0:
        return None
    greater = (ood_arr[:, None] > id_arr[None, :]).sum()
    ties = (ood_arr[:, None] == id_arr[None, :]).sum()
    return float((greater + 0.5 * ties) / (id_arr.size * ood_arr.size))


def selective_risk_metrics(confidence: Sequence[float], correct: Sequence[bool]) -> Dict[str, float]:
    """AURC and AUGRC from accepting samples in order of decreasing confidence.

    At coverage i/n, selective risk is errors/i among accepted samples and generalized risk is errors/n
    (accepted and wrong). Both areas are the mean over the n operating points.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    wrong = ~np.asarray(correct, dtype=bool)
    n = conf.size
    if n == 0:
        raise ValueError("selective_risk_metrics needs at least one sample")
    order = np.argsort(-conf, kind="stable")
    cumulative_errors = np.cumsum(wrong[order])
    accepted = np.arange(1, n + 1)
    return {
        "aurc": float(np.mean(cumulative_errors / accepted)),
        "augrc": float(np.mean(cumulative_errors / n)),
        "full_coverage_risk": float(cumulative_errors[-1] / n),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "accuracy": float((true == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def fit_linear_probe(train_features: np.ndarray, train_labels: np.ndarray, *, seed: int = 42) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=seed))
    probe.fit(train_features, train_labels)
    return probe


def evaluate_probe(
    probe: Any,
    *,
    train_features: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    ood_features: np.ndarray,
    knn_k: int = KNN_K,
) -> Dict[str, Any]:
    test_logits = probe.decision_function(test_features)
    ood_logits = probe.decision_function(ood_features)
    if test_logits.ndim == 1:  # binary LogisticRegression returns one margin per sample
        test_logits = np.stack([-test_logits, test_logits], axis=1) / 2.0
        ood_logits = np.stack([-ood_logits, ood_logits], axis=1) / 2.0
    predictions = test_logits.argmax(axis=1)
    test_scores = ood_scores_from_logits(test_logits)
    ood_scores = ood_scores_from_logits(ood_logits) if len(ood_features) else {key: np.empty(0) for key in test_scores}
    test_scores["knn"] = knn_distance(test_features, train_features, k=knn_k)
    ood_scores["knn"] = knn_distance(ood_features, train_features, k=knn_k) if len(ood_features) else np.empty(0)
    confidence = softmax(test_logits).max(axis=1)
    correct = predictions == np.asarray(test_labels)
    return {
        "classification": classification_metrics(test_labels, predictions),
        "ood_auroc": {name: ood_auroc(test_scores[name], ood_scores[name]) for name in test_scores},
        "selective": selective_risk_metrics(confidence, correct),
        "rows": {
            "test": [
                {"label": int(label), "prediction": int(pred), "confidence": float(conf)}
                | {f"score_{name}": float(values[index]) for name, values in test_scores.items()}
                for index, (label, pred, conf) in enumerate(zip(test_labels, predictions, confidence))
            ],
            "ood": [
                {f"score_{name}": float(values[index]) for name, values in ood_scores.items()}
                for index in range(len(ood_features))
            ],
        },
    }


def evaluate_adapter_rows(
    test_rows: Sequence[Dict[str, Any]],
    ood_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize per-sample adapter outputs (``label``, ``prediction``, ``confidence``, ``primary_score``)."""
    labels = [int(row["label"]) for row in test_rows]
    predictions = [int(row["prediction"]) for row in test_rows]
    correct = [label == pred for label, pred in zip(labels, predictions)]
    return {
        "classification": classification_metrics(labels, predictions),
        "ood_auroc": {
            "primary_score": ood_auroc(
                [row["primary_score"] for row in test_rows], [row["primary_score"] for row in ood_rows]
            )
        },
        "selective": selective_risk_metrics([row["confidence"] for row in test_rows], correct),
        "rows": {"test": list(test_rows), "ood": list(ood_rows)},
    }


def _iter_batches(loader: Iterable[Dict[str, Any]]) -> Iterable[tuple[Any, Any]]:
    for batch in loader:
        yield batch["images"], batch["labels"]


def extract_backbone_features(model: Any, loader: Any, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (CLS) features of the frozen backbone, fp32, as the adapters see them before LoRA."""
    import torch

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels in _iter_batches(loader):
            output = model(pixel_values=images.to(device))
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            features.append(pooled.float().cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(features), np.concatenate(labels)


def score_adapter(adapter: Any, loader: Any, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Run a released adapter over a loader; labels and predictions use the loader's class order."""
    index_by_name = {name: index for index, name in enumerate(class_names)}
    rows: List[Dict[str, Any]] = []
    for images, batch_labels in _iter_batches(loader):
        for label, payload in zip(batch_labels.tolist(), adapter.predict_batch_with_ood(images)):
            disease = dict(payload.get("disease") or {})
            ood = dict(payload.get("ood_analysis") or {})
            rows.append(
                {
                    "label": int(label),
                    "prediction": int(index_by_name.get(str(disease.get("name")), -1)),
                    "confidence": float(disease.get("confidence", 0.0)),
                    "primary_score": float(ood.get("primary_score", float("nan"))),
                    "is_ood": bool(ood.get("is_ood", False)),
                }
            )
    return rows


def _strip_rows(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key != "rows"}


def run(
    *,
    dataset_root: Path,
    target: str,
    output_dir: Path,
    adapter_dir: Optional[Path],
    device: str,
    backbone: str = DEFAULT_BACKBONE,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Dict[str, Any]:
    from transformers import AutoModel

    from src.data.loaders import create_training_loaders

    loaders = create_training_loaders(
        data_dir=str(dataset_root),
        crop=target,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler="shuffle",
        augmentation_policy="none",
        use_cache=False,
    )
    train_loader, test_loader = loaders["train"], loaders["test"]
    ood_loader = loaders.get("ood")
    class_names = list(getattr(train_loader.dataset, "classes", []) or [])

    model = AutoModel.from_pretrained(backbone).to(device).eval()
    train_x, train_y = extract_backbone_features(model, train_loader, device)
    test_x, test_y = extract_backbone_features(model, test_loader, device)
    ood_x, _ = extract_backbone_features(model, ood_loader, device) if ood_loader is not None else (np.empty((0, train_x.shape[1])), None)
    del model

    probe = fit_linear_probe(train_x, train_y)
    probe_result = evaluate_probe(
        probe, train_features=train_x, test_features=test_x, test_labels=test_y, ood_features=ood_x
    )

    adapter_result: Optional[Dict[str, Any]] = None
    if adapter_dir is not None:
        from src.adapter.independent_crop_adapter import IndependentCropAdapter

        crop, _, part = target.partition("__")
        adapter = IndependentCropAdapter(crop_name=crop, part_name=part or "unspecified", device=device)
        adapter.load_adapter(str(adapter_dir))
        adapter_result = evaluate_adapter_rows(
            score_adapter(adapter, test_loader, class_names),
            score_adapter(adapter, ood_loader, class_names) if ood_loader is not None else [],
        )

    summary = {
        "schema_version": "v1_posthoc_linear_probe",
        "post_hoc": True,
        "target": target,
        "backbone": backbone,
        "class_names": class_names,
        "sample_counts": {"train": int(len(train_y)), "test": int(len(test_y)), "ood": int(len(ood_x))},
        "probe": _strip_rows(probe_result),
        "adapter": _strip_rows(adapter_result) if adapter_result else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{target}--posthoc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = {"probe": probe_result["rows"], "adapter": adapter_result["rows"] if adapter_result else None}
    (output_dir / f"{target}--posthoc_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory that contains <target>/")
    parser.add_argument("--target", required=True, help="Runtime dataset target, for example tomato__leaf")
    parser.add_argument("--adapter-dir", type=Path, help="Released adapter bundle for the same target")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/posthoc_linear_probe"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    summary = run(
        dataset_root=args.dataset_root,
        target=args.target,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: summary[key] for key in ("target", "sample_counts", "probe", "adapter")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
