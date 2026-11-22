import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_CLI_HELP = """
import src.app.cli
sys.argv = ['cli', 'inference', '--help']
try:
    src.app.cli.main()
except SystemExit:
    pass
"""


@pytest.mark.parametrize(
    ("setup", "forbidden"),
    [
        ("import src.workflows.inference", ["src.training.continual_sd_lora", "peft", "sklearn"]),
        (_CLI_HELP, ["src.training.continual_sd_lora", "src.workflows.training"]),
    ],
    ids=["inference_workflow", "cli_inference_help"],
)
def test_inference_surfaces_avoid_training_stack(setup, forbidden):
    code = "\n".join(
        [
            "import json, sys",
            f"sys.path.insert(0, {str(ROOT)!r})",
            setup,
            f"print(json.dumps([name for name in {forbidden!r} if name in sys.modules]))",
        ]
    )
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    loaded = json.loads([line for line in process.stdout.splitlines() if line.strip()][-1])
    assert loaded == []
