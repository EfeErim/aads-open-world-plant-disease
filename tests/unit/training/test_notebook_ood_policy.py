from types import SimpleNamespace

from src.training.notebook_runtime_helpers import calibrate_notebook_ood_policy, select_notebook_ood_policy
from src.training.types import EvaluationArtifactsPayload, ValidationReport


class _Dataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class _Loader:
    def __init__(self, name: str, size: int = 4):
        self.name = name
        self.dataset = _Dataset(size)


class _Detector:
    def __init__(self):
        self.primary_score_method = "ensemble"
        self.threshold_overrides = {}

    def set_score_threshold_override(self, method: str, threshold: float) -> None:
        self.threshold_overrides[method] = threshold


class _Trainer:
    def __init__(self):
        self.config = SimpleNamespace(ood_primary_score_method="auto")
        self.ood_detector = _Detector()

    def set_ood_primary_score_method(self, method: str) -> str:
        self.config.ood_primary_score_method = method
        self.ood_detector.primary_score_method = method
        return method


def _evaluation() -> EvaluationArtifactsPayload:
    report = ValidationReport(
        val_loss=0.1,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy={"healthy": 1.0},
        per_class_support={"healthy": 4},
        worst_classes=[],
    )
    return EvaluationArtifactsPayload(
        report=report,
        y_true=[0, 0, 0, 0],
        y_pred=[0, 0, 0, 0],
        ood_labels=[0, 0, 0, 0, 1, 1, 1, 1],
        ood_scores=[0.8, 0.7, 0.6, 0.5, 0.55, 0.5, 0.45, 0.4],
        ood_scores_by_method={
            "ensemble": [0.8, 0.7, 0.6, 0.5, 0.55, 0.5, 0.45, 0.4],
            "energy": [0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3],
            "knn": [0.1, 0.2, 0.15, 0.18, 0.9, 0.8, 0.85, 0.75],
        },
    )


def test_select_notebook_ood_policy_uses_only_ood_dev_for_auto_selection(monkeypatch):
    trainer = _Trainer()
    val_loader = _Loader("val")
    ood_dev_loader = _Loader("ood_dev")
    ood_test_loader = _Loader("ood_test")
    calls = []

    def _evaluate(actual_trainer, id_loader, *, ood_loader):
        calls.append((actual_trainer, id_loader, ood_loader))
        return _evaluation()

    monkeypatch.setattr("src.training.validation.evaluate_model_with_artifact_metrics", _evaluate)

    result = select_notebook_ood_policy(
        trainer=trainer,
        loaders={"val": val_loader, "ood_dev": ood_dev_loader, "ood": ood_test_loader},
        continual_config={"ood": {"primary_score_method": "auto", "real_dev_target_fpr": 0.05}},
    )

    assert calls == [(trainer, val_loader, ood_dev_loader)]
    assert result["selection_source"] == "real_ood_dev"
    assert result["selected_primary_score_method"] == "knn"
    assert result["selected_threshold"] is not None
    assert trainer.ood_detector.primary_score_method == "knn"
    assert trainer.ood_detector.threshold_overrides["knn"] == result["selected_threshold"]


def test_select_notebook_ood_policy_does_not_tune_on_final_ood_without_dev(monkeypatch):
    trainer = _Trainer()

    def _unexpected_evaluation(*_args, **_kwargs):
        raise AssertionError("final OOD evidence must not be used for selection")

    monkeypatch.setattr("src.training.validation.evaluate_model_with_artifact_metrics", _unexpected_evaluation)

    result = select_notebook_ood_policy(
        trainer=trainer,
        loaders={"val": _Loader("val"), "ood": _Loader("ood_test")},
        continual_config={"ood": {"primary_score_method": "auto"}},
    )

    assert result["selection_source"] == "real_ood_guardrail_no_dev"
    assert result["selected_primary_score_method"] == "ensemble"
    assert result["selected_threshold"] is None


def test_calibrate_notebook_ood_policy_records_selection_in_export_metadata(monkeypatch):
    class _Adapter:
        def __init__(self):
            self._trainer = _Trainer()
            self.calibration_loader = None
            self.export_metadata = None

        def calibrate_ood(self, loader):
            self.calibration_loader = loader
            return {"ood_calibration": {"version": 4, "num_classes": 2}}

        def set_export_metadata(self, *, ood_calibration):
            self.export_metadata = ood_calibration

    adapter = _Adapter()
    val_loader = _Loader("val")
    ood_dev_loader = _Loader("ood_dev")
    monkeypatch.setattr(
        "src.training.validation.evaluate_model_with_artifact_metrics",
        lambda _trainer, _id_loader, *, ood_loader: _evaluation(),
    )

    result = calibrate_notebook_ood_policy(
        adapter=adapter,
        loaders={"val": val_loader, "ood_dev": ood_dev_loader, "ood": _Loader("ood_test")},
        continual_config={"ood": {"primary_score_method": "auto"}},
    )

    calibration = result["calibration"]["ood_calibration"]
    assert adapter.calibration_loader is val_loader
    assert calibration["requested_primary_score_method"] == "auto"
    assert calibration["primary_score_method"] == "knn"
    assert calibration["selection_source"] == "real_ood_dev"
    assert calibration["selected_threshold"] == result["policy"]["selected_threshold"]
    assert adapter.export_metadata == calibration
