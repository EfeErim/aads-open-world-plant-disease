# Auto-extracted from colab_notebooks/1_identify_crop_part_with_router.ipynb cell 5.
# Keep notebook execute-only cells thin; edit behavior here.

import json

from scripts.colab_repo_bootstrap import login_and_check_hf_token
from scripts.colab_router_adapter_inference import clear_router_cache, ensure_router_ready, run_inference

# Notebook 1 user-facing profile controls
INFERENCE_PROFILES = {
    'custom': {},
    'a100_colab_default': {
        'CONFIG_ENV': 'colab',
        'DEVICE': 'cuda',
        'REQUIRE_HF_LOGIN': False,
        'CROP_HINT': None,
        'PART_HINT': None,
        'FORCE_UPLOAD_IF_NO_IMAGE': True,
        # Balanced is recommended for safer crop selection on diseased images.
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

SPEED_PROFILES = {
    'fast': {
        'SHOW_ROUTER_DIAGNOSTICS': False,
        'TOP_CROP_CANDIDATES': 2,
        'MAX_IMAGE_SIDE_FOR_ROUTER': 1280,
        'RENDER_ROUTER_VISUALIZATION': False,
        'MAX_RENDERED_BOXES': 0,
        'INCLUDE_ADAPTER_TARGET': True,
    },
    'balanced': {
        'SHOW_ROUTER_DIAGNOSTICS': True,
        'TOP_CROP_CANDIDATES': 3,
        'MAX_IMAGE_SIDE_FOR_ROUTER': 1600,
        'RENDER_ROUTER_VISUALIZATION': True,
        'MAX_RENDERED_BOXES': 6,
        'INCLUDE_ADAPTER_TARGET': True,
    },
    'quality': {
        'SHOW_ROUTER_DIAGNOSTICS': True,
        'TOP_CROP_CANDIDATES': 5,
        'MAX_IMAGE_SIDE_FOR_ROUTER': None,
        'RENDER_ROUTER_VISUALIZATION': True,
        'MAX_RENDERED_BOXES': 0,
        'INCLUDE_ADAPTER_TARGET': True,
    },
}

CONFIG_ENV = 'colab'
DEVICE = 'cuda'
REQUIRE_HF_LOGIN = False
CROP_HINT = None
PART_HINT = None
IMAGE_PATH = None  # Optional default image path
FORCE_UPLOAD_IF_NO_IMAGE = True
PRELOAD_ROUTER = True
RESET_ROUTER_CACHE = False
ROUTER_RUNTIME_PROFILE = 'balanced'

# Diagnostic output for monitoring router decision quality (recommended: True).
SHOW_ROUTER_DIAGNOSTICS = True
TOP_CROP_CANDIDATES = 3
PRINT_JSON_RESULT = False

# Speed settings
MAX_IMAGE_SIDE_FOR_ROUTER = None  # Ornek: 1280 veya 1600
RENDER_ROUTER_VISUALIZATION = True
MAX_RENDERED_BOXES = 0  # 0 = sinirsiz
INCLUDE_ADAPTER_TARGET = True

# Notebook safety gate: uncertain router-only predictions become uncertain/unknown.
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

speed_profile = dict(SPEED_PROFILES.get(NOTEBOOK_SPEED_MODE, {}))
SHOW_ROUTER_DIAGNOSTICS = bool(speed_profile.get('SHOW_ROUTER_DIAGNOSTICS', SHOW_ROUTER_DIAGNOSTICS))
TOP_CROP_CANDIDATES = int(speed_profile.get('TOP_CROP_CANDIDATES', TOP_CROP_CANDIDATES))
MAX_IMAGE_SIDE_FOR_ROUTER = speed_profile.get('MAX_IMAGE_SIDE_FOR_ROUTER', MAX_IMAGE_SIDE_FOR_ROUTER)
RENDER_ROUTER_VISUALIZATION = bool(speed_profile.get('RENDER_ROUTER_VISUALIZATION', RENDER_ROUTER_VISUALIZATION))
MAX_RENDERED_BOXES = int(speed_profile.get('MAX_RENDERED_BOXES', MAX_RENDERED_BOXES))
INCLUDE_ADAPTER_TARGET = bool(speed_profile.get('INCLUDE_ADAPTER_TARGET', INCLUDE_ADAPTER_TARGET))
ENFORCE_NOTEBOOK_ROUTER_GATE = bool(speed_profile.get('ENFORCE_NOTEBOOK_ROUTER_GATE', ENFORCE_NOTEBOOK_ROUTER_GATE))

# Gerekirse son override katmani.
INFERENCE_OVERRIDES = dict(globals().get('INFERENCE_OVERRIDES') or {})
if INFERENCE_OVERRIDES:
    CONFIG_ENV = str(INFERENCE_OVERRIDES.get('CONFIG_ENV', CONFIG_ENV))
    DEVICE = str(INFERENCE_OVERRIDES.get('DEVICE', DEVICE))
    REQUIRE_HF_LOGIN = bool(INFERENCE_OVERRIDES.get('REQUIRE_HF_LOGIN', REQUIRE_HF_LOGIN))
    CROP_HINT = INFERENCE_OVERRIDES.get('CROP_HINT', CROP_HINT)
    PART_HINT = INFERENCE_OVERRIDES.get('PART_HINT', PART_HINT)
    IMAGE_PATH = INFERENCE_OVERRIDES.get('IMAGE_PATH', IMAGE_PATH)
    FORCE_UPLOAD_IF_NO_IMAGE = bool(INFERENCE_OVERRIDES.get('FORCE_UPLOAD_IF_NO_IMAGE', FORCE_UPLOAD_IF_NO_IMAGE))
    PRELOAD_ROUTER = bool(INFERENCE_OVERRIDES.get('PRELOAD_ROUTER', PRELOAD_ROUTER))
    RESET_ROUTER_CACHE = bool(INFERENCE_OVERRIDES.get('RESET_ROUTER_CACHE', RESET_ROUTER_CACHE))
    ROUTER_RUNTIME_PROFILE = str(INFERENCE_OVERRIDES.get('ROUTER_RUNTIME_PROFILE', ROUTER_RUNTIME_PROFILE) or '').strip() or None
    SHOW_ROUTER_DIAGNOSTICS = bool(INFERENCE_OVERRIDES.get('SHOW_ROUTER_DIAGNOSTICS', SHOW_ROUTER_DIAGNOSTICS))
    TOP_CROP_CANDIDATES = int(INFERENCE_OVERRIDES.get('TOP_CROP_CANDIDATES', TOP_CROP_CANDIDATES))
    PRINT_JSON_RESULT = bool(INFERENCE_OVERRIDES.get('PRINT_JSON_RESULT', PRINT_JSON_RESULT))
    MAX_IMAGE_SIDE_FOR_ROUTER = INFERENCE_OVERRIDES.get('MAX_IMAGE_SIDE_FOR_ROUTER', MAX_IMAGE_SIDE_FOR_ROUTER)
    RENDER_ROUTER_VISUALIZATION = bool(
        INFERENCE_OVERRIDES.get('RENDER_ROUTER_VISUALIZATION', RENDER_ROUTER_VISUALIZATION)
    )
    MAX_RENDERED_BOXES = int(INFERENCE_OVERRIDES.get('MAX_RENDERED_BOXES', MAX_RENDERED_BOXES))
    INCLUDE_ADAPTER_TARGET = bool(INFERENCE_OVERRIDES.get('INCLUDE_ADAPTER_TARGET', INCLUDE_ADAPTER_TARGET))
    ENFORCE_NOTEBOOK_ROUTER_GATE = bool(
        INFERENCE_OVERRIDES.get('ENFORCE_NOTEBOOK_ROUTER_GATE', ENFORCE_NOTEBOOK_ROUTER_GATE)
    )
    ROUTER_RESULT_MIN_CONFIDENCE = float(
        INFERENCE_OVERRIDES.get('ROUTER_RESULT_MIN_CONFIDENCE', ROUTER_RESULT_MIN_CONFIDENCE)
    )
    ROUTER_RESULT_MIN_MARGIN = float(
        INFERENCE_OVERRIDES.get('ROUTER_RESULT_MIN_MARGIN', ROUTER_RESULT_MIN_MARGIN)
    )

print(
    f'[SETTINGS] profile={NOTEBOOK_PROFILE} speed_mode={NOTEBOOK_SPEED_MODE} env={CONFIG_ENV} device={DEVICE} '
    f'crop_hint={CROP_HINT} part_hint={PART_HINT} preload_router={PRELOAD_ROUTER} '
    f'router_runtime_profile={ROUTER_RUNTIME_PROFILE} diagnostics={SHOW_ROUTER_DIAGNOSTICS} top_candidates={TOP_CROP_CANDIDATES} '
    f'max_side={MAX_IMAGE_SIDE_FOR_ROUTER} render_visual={RENDER_ROUTER_VISUALIZATION} max_boxes={MAX_RENDERED_BOXES} '
    f'include_adapter_target={INCLUDE_ADAPTER_TARGET} '
    f'notebook_gate={ENFORCE_NOTEBOOK_ROUTER_GATE} min_conf={ROUTER_RESULT_MIN_CONFIDENCE:.2f} min_margin={ROUTER_RESULT_MIN_MARGIN:.2f}'
)
print('[NEXT] After router setup completes, run the analysis cell. For a new image, rerun only the analysis cell.')

HF_READY = False
if REQUIRE_HF_LOGIN:
    HF_READY = login_and_check_hf_token(print_fn=print)
    if not HF_READY:
        raise RuntimeError('Notebook settings require Hugging Face login, but authentication failed.')
else:
    print('[HF] Required login is disabled. Add a Colab secret if a gated model requires access.')

if RESET_ROUTER_CACHE:
    clear_router_cache()
    print('[ROUTER] Session cache cleared.')

if CROP_HINT:
    print('[ROUTER] crop_hint provided; router will be skipped and preload was not requested.')
elif PRELOAD_ROUTER:
    ensure_router_ready(
        config_env=CONFIG_ENV,
        device=DEVICE,
        status_printer=print,
        runtime_profile=ROUTER_RUNTIME_PROFILE,
    )
    print('[NOTEBOOK] Router is ready for this runtime. Rerun the next cell with a new image when needed.')
else:
    print('[NOTEBOOK] PRELOAD_ROUTER is disabled. The first analysis call will load the router.')
