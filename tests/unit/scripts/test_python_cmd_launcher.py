import subprocess
from pathlib import Path


def test_python_cmd_propagates_success_exit_code():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [str(repo_root / "scripts" / "python.cmd"), "-c", "import sys; sys.exit(0)"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


def test_python_cmd_propagates_nonzero_exit_code():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [str(repo_root / "scripts" / "python.cmd"), "-c", "import sys; sys.exit(7)"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
