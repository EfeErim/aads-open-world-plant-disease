from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_adapter_metadata_completeness import validate_adapter


def test_validate_adapter_accepts_deployed_pth_classifier_and_colocated_behavioral_artifact(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "tomato" / "leaf" / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "v6",
                "class_to_idx": {"healthy": 0},
                "backbone": {"model_name": "test"},
                "ood_calibration": {"version": 1},
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter_dir / "classifier.pth").write_bytes(b"state")
    (adapter_dir / "adapter_behavioral_acceptance.json").write_text('{"status":"pass"}', encoding="utf-8")
    result = validate_adapter(adapter_dir, {"adapter_dir": str(adapter_dir), "crop_name": "tomato", "part_name": "leaf"})
    assert result["status"] == "pass"
    assert result["behavioral_acceptance_status"] == "pass"
