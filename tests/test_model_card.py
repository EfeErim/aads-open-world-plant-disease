from __future__ import annotations

import json
from pathlib import Path

import pytest

from aads_public.model_card import build_target_model_card


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "passed": False,
            "deployable": False,
            "missing_requirements": ["ood_false_positive_rate"],
            "classification_evidence": {
                "metrics": {
                    "accuracy": 0.9,
                    "macro_f1": 0.8,
                    "ood_auroc": 0.7,
                    "ood_false_positive_rate": 0.4,
                    "classification_samples": 100,
                    "ood_samples": 40,
                }
            },
        },
    )
    config = _write_json(
        tmp_path / "config.json",
        {"base_model_name_or_path": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
    )
    meta = _write_json(
        tmp_path / "meta.json",
        {"crop_name": "apricot", "part_name": "fruit"},
    )
    return readiness, config, meta


def test_model_card_states_failed_evidence_and_no_placeholders(tmp_path: Path) -> None:
    card = build_target_model_card(*_inputs(tmp_path), target="apricot__fruit")

    assert "Production readiness | **FAILED**" in card
    assert "Deployable | **No**" in card
    assert "OOD false-positive rate | 0.400" in card
    assert "[More Information Needed]" not in card
    assert "autonomous diagnosis" in card


def test_model_card_rejects_target_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_target_model_card(*_inputs(tmp_path), target="grape__fruit")


def test_model_card_rejects_ready_artifact(tmp_path: Path) -> None:
    readiness, config, meta = _inputs(tmp_path)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["passed"] = True
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly failed"):
        build_target_model_card(readiness, config, meta, target="apricot__fruit")
