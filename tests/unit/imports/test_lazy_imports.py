import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run_probe(code: str) -> dict[str, bool]:
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_lines = [line for line in process.stdout.splitlines() if line.strip()]
    return json.loads(stdout_lines[-1])


def test_importing_pipeline_package_is_lazy():
    probe = _run_probe(
        "\n".join(
            [
                "import json, sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import src.pipeline",
                "print(json.dumps({'router_runtime_loaded': 'src.pipeline.router_adapter_runtime' in sys.modules}))",
            ]
        )
    )
    assert probe["router_runtime_loaded"] is False


def test_importing_workflows_package_is_lazy():
    probe = _run_probe(
        "\n".join(
            [
                "import json, sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import src.workflows",
                "print(json.dumps({"
                "'inference_loaded': 'src.workflows.inference' in sys.modules, "
                "'training_loaded': 'src.workflows.training' in sys.modules"
                "}))",
            ]
        )
    )
    assert probe["inference_loaded"] is False
    assert probe["training_loaded"] is False


def test_importing_inference_workflow_avoids_training_stack():
    probe = _run_probe(
        "\n".join(
            [
                "import json, sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import src.workflows.inference",
                "print(json.dumps({"
                "'trainer_loaded': 'src.training.continual_sd_lora' in sys.modules, "
                "'peft_loaded': 'peft' in sys.modules"
                "}))",
            ]
        )
    )
    assert probe["trainer_loaded"] is False
    assert probe["peft_loaded"] is False


def test_cli_inference_help_avoids_training_stack():
    probe = _run_probe(
        "\n".join(
            [
                "import json, sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import src.app.cli",
                "sys.argv = ['cli', 'inference', '--help']",
                "try:",
                "    src.app.cli.main()",
                "except SystemExit:",
                "    pass",
                "print(json.dumps({"
                "'trainer_loaded': 'src.training.continual_sd_lora' in sys.modules, "
                "'workflow_training_loaded': 'src.workflows.training' in sys.modules"
                "}))",
            ]
        )
    )
    assert probe["trainer_loaded"] is False
    assert probe["workflow_training_loaded"] is False
