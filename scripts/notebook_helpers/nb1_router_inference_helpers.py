from __future__ import annotations
from pathlib import Path
from src.core.config_manager import get_config
from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report, login_and_check_hf_token, running_in_colab
from scripts.colab_router_adapter_inference import clear_router_cache, ensure_router_ready, run_inference


def run_bootstrap_notebook_nb1(notebook_name: str = "Notebook 1: Router Adapter Inference", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 1 with repo discovery and access checks."""
    from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status
    try:
        from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
        repo_root_for_update_check = _ensure_repo_root_for_update_check()
    except Exception:
        repo_root_for_update_check = None

    # [KONTROL] Ilk hucre: Bootstrap kontrati kontrol edildi
    BOOTSTRAP = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    ROOT = BOOTSTRAP["ROOT"]
    print_bootstrap_status(BOOTSTRAP)
    return BOOTSTRAP


def run_access_check_nb1(ROOT: Path, print_fn=print) -> dict:
    """Check model access for router inference."""
    CONFIG_FOR_ACCESS = get_config(environment='colab')
    ROUTER_VLM_CFG = dict(dict(CONFIG_FOR_ACCESS.get('router', {})).get('vlm', {}))
    ROUTER_MODEL_IDS = [
        str(model_id).strip()
        for model_id in list(dict(ROUTER_VLM_CFG.get('model_ids', {})).values())
        if str(model_id).strip()
    ]

    print_fn("[SETUP] Checking model access for router models...")
    ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=ROUTER_MODEL_IDS)
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    return ACCESS_REPORT


def init_inference_config_nb1() -> dict:
    """Initialize inference configuration profiles and parameters."""
    NOTEBOOK_PROFILE = 'a100_colab_default'

    INFERENCE_PROFILES = {
        'custom': {},
        'a100_colab_default': {
            'CONFIG_ENV': 'colab',
            'DEVICE': 'cuda',
            'REQUIRE_HF_LOGIN': False,
            'CROP_HINT': None,
            'PART_HINT': None,
            'FORCE_UPLOAD_IF_NO_IMAGE': True,
            'ROUTER_RUNTIME_PROFILE': 'balanced',
        },
        'cpu_debug': {
            'CONFIG_ENV': 'colab',
            'DEVICE': 'cpu',
            'REQUIRE_HF_LOGIN': False,
            'CROP_HINT': None,
            'PART_HINT': None,
            'FORCE_UPLOAD_IF_NO_IMAGE': False,
            'ROUTER_RUNTIME_PROFILE': 'balanced',
        },
    }

    CONFIG_ENV = 'colab'
    DEVICE = 'cuda'
    REQUIRE_HF_LOGIN = False
    CROP_HINT = None
    PART_HINT = None
    IMAGE_PATH = None
    FORCE_UPLOAD_IF_NO_IMAGE = True
    PRELOAD_ROUTER = True
    RESET_ROUTER_CACHE = False
    ROUTER_RUNTIME_PROFILE = 'balanced'
    SHOW_ROUTER_DIAGNOSTICS = True
    TOP_CROP_CANDIDATES = 3
    PRINT_JSON_RESULT = False
    ENFORCE_NOTEBOOK_ROUTER_GATE = True
    ROUTER_RESULT_MIN_CONFIDENCE = 0.65
    ROUTER_RESULT_MIN_MARGIN = 0.10

    profile = dict(INFERENCE_PROFILES.get(NOTEBOOK_PROFILE, {}))
    CONFIG_ENV = str(profile.get('CONFIG_ENV', CONFIG_ENV))
    DEVICE = str(profile.get('DEVICE', DEVICE))
    REQUIRE_HF_LOGIN = bool(profile.get('REQUIRE_HF_LOGIN', REQUIRE_HF_LOGIN))
    CROP_HINT = profile.get('CROP_HINT', CROP_HINT)
    PART_HINT = profile.get('PART_HINT', PART_HINT)
    FORCE_UPLOAD_IF_NO_IMAGE = bool(profile.get('FORCE_UPLOAD_IF_NO_IMAGE', FORCE_UPLOAD_IF_NO_IMAGE))
    PRELOAD_ROUTER = bool(profile.get('PRELOAD_ROUTER', PRELOAD_ROUTER))
    RESET_ROUTER_CACHE = bool(profile.get('RESET_ROUTER_CACHE', RESET_ROUTER_CACHE))
    ROUTER_RUNTIME_PROFILE = str(profile.get('ROUTER_RUNTIME_PROFILE', ROUTER_RUNTIME_PROFILE) or '').strip() or None

    return {
        'NOTEBOOK_PROFILE': NOTEBOOK_PROFILE,
        'INFERENCE_PROFILES': INFERENCE_PROFILES,
        'CONFIG_ENV': CONFIG_ENV,
        'DEVICE': DEVICE,
        'REQUIRE_HF_LOGIN': REQUIRE_HF_LOGIN,
        'CROP_HINT': CROP_HINT,
        'PART_HINT': PART_HINT,
        'IMAGE_PATH': IMAGE_PATH,
        'FORCE_UPLOAD_IF_NO_IMAGE': FORCE_UPLOAD_IF_NO_IMAGE,
        'PRELOAD_ROUTER': PRELOAD_ROUTER,
        'RESET_ROUTER_CACHE': RESET_ROUTER_CACHE,
        'ROUTER_RUNTIME_PROFILE': ROUTER_RUNTIME_PROFILE,
        'SHOW_ROUTER_DIAGNOSTICS': SHOW_ROUTER_DIAGNOSTICS,
        'TOP_CROP_CANDIDATES': TOP_CROP_CANDIDATES,
        'PRINT_JSON_RESULT': PRINT_JSON_RESULT,
        'ENFORCE_NOTEBOOK_ROUTER_GATE': ENFORCE_NOTEBOOK_ROUTER_GATE,
        'ROUTER_RESULT_MIN_CONFIDENCE': ROUTER_RESULT_MIN_CONFIDENCE,
        'ROUTER_RESULT_MIN_MARGIN': ROUTER_RESULT_MIN_MARGIN,
    }


def setup_inference_nb1(config: dict, print_fn=print) -> bool:
    """Setup HuggingFace login and preload router if needed."""
    HF_READY = False
    if config['REQUIRE_HF_LOGIN']:
        HF_READY = login_and_check_hf_token(print_fn=print_fn)
        if not HF_READY:
            raise RuntimeError('Notebook ayari HF login istiyor ama dogrulama basarisiz oldu.')
    else:
        print_fn('[HF] Zorunlu login kapali. Gated model gerekiyorsa Colab secret ekleyin.')

    if config['RESET_ROUTER_CACHE']:
        clear_router_cache()
        print_fn('[ROUTER] Oturum cache temizlendi.')

    if config['CROP_HINT']:
        print_fn('[ROUTER] crop_hint verildi; router atlanacak, on-yukleme yapilmadi.')
    elif config['PRELOAD_ROUTER']:
        ensure_router_ready(
            config_env=config['CONFIG_ENV'],
            device=config['DEVICE'],
            status_printer=print_fn,
            runtime_profile=config['ROUTER_RUNTIME_PROFILE'],
        )
        print_fn('[NOTEBOOK] Router bu runtime icin hazir. Sonraki hucreyi yeni bir goruntu ile tekrar calistirabilirsiniz.')
    else:
        print_fn('[NOTEBOOK] PRELOAD_ROUTER kapali. Ilk analiz cagrisi routeri bu sirada yukleyecek.')

    return HF_READY


__all__ = [
    'run_bootstrap_notebook_nb1',
    'run_access_check_nb1',
    'init_inference_config_nb1',
    'setup_inference_nb1',
]
