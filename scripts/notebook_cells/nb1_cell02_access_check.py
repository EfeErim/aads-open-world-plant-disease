# Auto-extracted from colab_notebooks/1_identify_crop_part_with_router.ipynb cell 2.
# Keep notebook execute-only cells thin; edit behavior here.

from src.core.config_manager import get_config
from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report

CONFIG_FOR_ACCESS = get_config(environment='colab')
ROUTER_VLM_CFG = dict(dict(CONFIG_FOR_ACCESS.get('router', {})).get('vlm', {}))
ROUTER_MODEL_IDS = [
    str(model_id).strip()
    for model_id in list(dict(ROUTER_VLM_CFG.get('model_ids', {})).values())
    if str(model_id).strip()
]
ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=ROUTER_MODEL_IDS)
print_notebook_access_report(ACCESS_REPORT, print_fn=print)
