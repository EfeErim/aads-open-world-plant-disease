# Auto-extracted from colab_notebooks/8_auto_router_adapter_prediction.ipynb deployment-release setup.
# Keep notebook execute-only cells thin; edit behavior here.

import os
from pathlib import Path

from src.core.config_manager import get_config
from src.pipeline.adapter_release import prepare_notebook_adapter_root

cell_script_root = Path(str(globals().get("__notebook_cell_script_root__", ""))).resolve()
repo_root = cell_script_root.parents[1] if cell_script_root.name == "notebook_cells" else Path.cwd().resolve()
config_env = str(globals().get("CONFIG_ENV", "colab"))
release_manifest_path = repo_root / "docs/evidence/current/demo_release/release_manifest.json"
os.environ.setdefault("GITHUB_RELEASE_REPOSITORY", "EfeErim/bitirmeprojesi")

if not os.environ.get("AADS_GITHUB_RELEASE_READ_TOKEN"):
    try:
        from google.colab import userdata

        release_read_token = str(userdata.get("AADS_GITHUB_RELEASE_READ_TOKEN") or "").strip()
    except Exception:  # Colab secret access raises provider-specific exceptions.
        release_read_token = ""
    if release_read_token:
        os.environ["AADS_GITHUB_RELEASE_READ_TOKEN"] = release_read_token

adapter_release_state = prepare_notebook_adapter_root(
    get_config(environment=config_env),
    repo_root,
    manifest_path=release_manifest_path,
)
ADAPTER_ROOT = adapter_release_state["adapter_root"]
if adapter_release_state["deployment_release"]:
    print(
        "[ADAPTER_RELEASE] "
        f"release_tag={adapter_release_state['release_tag']} "
        f"release_id={adapter_release_state['release_id']} "
        f"fetched={adapter_release_state['fetched']} verified={adapter_release_state['verified']}"
    )
else:
    print(f"[ADAPTER_RELEASE] legacy adapter root retained: {ADAPTER_ROOT}")
