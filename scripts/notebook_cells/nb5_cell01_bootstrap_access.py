# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper
import os
import subprocess
from pathlib import Path


def _configure_colab_git_read_access() -> None:
    token = str(os.environ.get("AADS_GITHUB_RELEASE_READ_TOKEN", "")).strip()
    if not token:
        try:
            from google.colab import userdata

            token = str(userdata.get("AADS_GITHUB_RELEASE_READ_TOKEN") or "").strip()
        except Exception:  # Colab secret access raises provider-specific exceptions.
            token = ""
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    if not token:
        return
    askpass = Path("/tmp/aads_git_read_askpass.sh")
    askpass.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "*Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "*) printf '%s\\n' \"$AADS_GIT_READ_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    os.environ["GIT_ASKPASS"] = str(askpass)
    os.environ["GIT_ASKPASS_REQUIRE"] = "force"
    os.environ["AADS_GIT_READ_TOKEN"] = token


_configure_colab_git_read_access()


CLONE_TARGET = Path('/content/bitirmeprojesi')  # Colab GitHub bootstrap contract
REPO_URL = os.environ.get('AADS_REPO_URL', 'https://github.com/EfeErim/aads-open-world-plant-disease.git')

print('[BOOTSTRAP] Notebook 5 bootstrap cell started.', flush=True)
print(f'[BOOTSTRAP] cwd={Path.cwd()}', flush=True)
print(f'[BOOTSTRAP] clone_target={CLONE_TARGET}', flush=True)

# Git clone if needed
if not CLONE_TARGET.exists():
    clone_url = REPO_URL
    print(f'[BOOTSTRAP] cloning repo from {REPO_URL}', flush=True)
    subprocess.run(['git', 'clone', '--depth', '1', '--progress', clone_url, str(CLONE_TARGET)], check=True)
else:
    print('[BOOTSTRAP] clone target already exists.', flush=True)

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb5_calibration_helpers import (  # noqa: E402
    run_access_check_nb5,
    run_bootstrap_notebook_nb5,
)

BOOTSTRAP = run_bootstrap_notebook_nb5()
ROOT = BOOTSTRAP["ROOT"]
print(f'[BOOTSTRAP] resolved ROOT={ROOT}', flush=True)

# Check model access
ACCESS_REPORT = run_access_check_nb5(ROOT, print_fn=print)
