import json
import sys
from pathlib import Path

import src.app.cli as cli


class _FakeImage:
    def __init__(self) -> None:
        self.convert_mode = ""

    def convert(self, mode: str):
        self.convert_mode = str(mode)
        return self


def test_cli_inference_dispatches_workflow(monkeypatch, capsys, tmp_path: Path):
    import src.workflows.inference as inference_module

    calls: dict[str, object] = {}
    fake_image = _FakeImage()

    class FakeWorkflow:
        def __init__(self, *, environment, device, adapter_root):
            calls["init"] = {
                "environment": environment,
                "device": device,
                "adapter_root": adapter_root,
            }

        def predict(self, image, *, crop_hint=None, part_hint=None, return_ood=True, trust_crop_hint=False):
            calls["predict"] = {
                "image": image,
                "crop_hint": crop_hint,
                "part_hint": part_hint,
                "return_ood": return_ood,
                "trust_crop_hint": trust_crop_hint,
            }
            return {
                "status": "success",
                "diagnosis": "healthy",
                "crop": "tomato",
                "router": {
                    "status": "skipped",
                    "message": "Router skipped because crop_hint was provided.",
                    "detections_count": 1,
                    "primary_detection": {
                        "crop": "tomato",
                        "part": "leaf",
                        "crop_confidence": 1.0,
                        "part_confidence": 1.0,
                    },
                },
            }

    def _open_image(path: Path):
        calls["opened_image"] = Path(path)
        return fake_image

    monkeypatch.setattr(inference_module, "InferenceWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli.Image, "open", _open_image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "inference",
            str(tmp_path / "leaf.png"),
            "--config-env",
            "colab",
            "--crop",
            "tomato",
            "--part",
            "leaf",
            "--adapter-root",
            str(tmp_path / "models" / "adapters"),
            "--device",
            "cpu",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert calls["opened_image"] == tmp_path / "leaf.png"
    assert fake_image.convert_mode == "RGB"
    assert calls["init"] == {
        "environment": "colab",
        "device": "cpu",
        "adapter_root": tmp_path / "models" / "adapters",
    }
    assert calls["predict"] == {
        "image": fake_image,
        "crop_hint": "tomato",
        "part_hint": "leaf",
        "return_ood": True,
        "trust_crop_hint": False,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "success",
        "diagnosis": "healthy",
        "crop": "tomato",
        "router": {
            "status": "skipped",
            "message": "Router skipped because crop_hint was provided.",
            "detections_count": 1,
            "primary_detection": {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 1.0,
                "part_confidence": 1.0,
            },
        },
    }


def test_cli_training_dispatches_workflow(monkeypatch, capsys, tmp_path: Path):
    import src.workflows.training as training_module

    calls: dict[str, object] = {}

    class FakeResult:
        def to_dict(self):
            return {"run_id": "run_cli", "crop_name": "tomato", "status": "completed"}

    class FakeWorkflow:
        def __init__(self, *, environment, device):
            calls["init"] = {"environment": environment, "device": device}

        def run(
            self,
            *,
            crop_name,
            data_dir,
            output_dir,
            num_epochs=None,
            num_workers=None,
            validation_every_n_epochs=None,
            part_name=None,
        ):
            calls["run"] = {
                "crop_name": crop_name,
                "data_dir": data_dir,
                "output_dir": output_dir,
                "num_epochs": num_epochs,
                "num_workers": num_workers,
                "validation_every_n_epochs": validation_every_n_epochs,
                "part_name": part_name,
            }
            return FakeResult()

    monkeypatch.setattr(training_module, "TrainingWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "training",
            "tomato",
            str(tmp_path / "runtime_data"),
            str(tmp_path / "outputs"),
            "--config-env",
            "colab",
            "--device",
            "cpu",
            "--num-epochs",
            "3",
            "--num-workers",
            "2",
            "--validation-every-n-epochs",
            "4",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert calls["init"] == {"environment": "colab", "device": "cpu"}
    assert calls["run"] == {
        "crop_name": "tomato",
        "data_dir": tmp_path / "runtime_data",
        "output_dir": tmp_path / "outputs",
        "num_epochs": 3,
        "num_workers": 2,
        "validation_every_n_epochs": 4,
        "part_name": None,
    }
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "run_cli",
        "crop_name": "tomato",
        "status": "completed",
    }
