# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 4.
# Resolves Notebook 2 parameters after the user-editable values are set.

# User-facing Notebook 2 adapter/default surface lives in the notebook cell.
# Keep behavior here aligned with visible ADAPTER_RECS and DEFAULT_RUNTIME_PARAMS.
# Notebook 2 calisma kimligi
# Bootstrap/validator defaultlari; ADAPTER_KEY secimi asagida bunlari kesin olarak ezer.
CROP_NAME = "tomato"
PART_NAME = "unspecified"
DATASET_NAME = ""

# Tek degistirilecek alan: egitmek istedigin adapter anahtari.
# Bu anahtar crop/part, runtime dataset, OOD ve OE yollarini gorunur sekilde set eder.
ADAPTER_KEY = str(globals().get("ADAPTER_KEY", "grape__fruit")).strip()
_USER_MANUAL_PARAM_OVERRIDES = dict(globals().get("MANUAL_PARAM_OVERRIDES") or {})
_USER_DEFAULT_RUNTIME_PARAMS = dict(globals().get("DEFAULT_RUNTIME_PARAMS") or {})

try:
    from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs
except Exception:
    _SHARED_ADAPTER_RECS = None
else:
    _SHARED_ADAPTER_RECS = get_adapter_recs()

ADAPTER_RECS = dict(globals().get("ADAPTER_RECS") or _SHARED_ADAPTER_RECS or {
    "grape__fruit": {
        "crop": "grape", "part": "fruit",
        "ood": "grape__fruit/ood",
        "oe": "grape__fruit/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 32, "BATCH_SIZE": 48, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.16, "OOD_FACTOR": 3.0, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.24, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.14, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.20},
    },
    "grape__leaf": {
        "crop": "grape", "part": "leaf",
        "ood": "grape__leaf/ood",
        "oe": "grape__leaf/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 56, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.14, "OOD_FACTOR": 3.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.24, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.15},
    },
    "strawberry__fruit": {
        "crop": "strawberry", "part": "fruit",
        "ood": "strawberry__fruit/ood",
        "oe": "strawberry__fruit/oe",
        "oe_enabled": True, "oe_w": 0.12, "allow_under_min": False,
        "defaults": {"EPOCHS": 38, "BATCH_SIZE": 40, "LEARNING_RATE": 5e-5, "LORA_R": 20, "LORA_ALPHA": 26, "LORA_DROPOUT": 0.13, "OOD_FACTOR": 2.0, "LABEL_SMOOTHING": 0.05, "OE_LOSS_WEIGHT": 0.12, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.99, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "strawberry__leaf": {
        "crop": "strawberry", "part": "leaf",
        "ood": "strawberry__leaf/ood",
        "oe": "strawberry__leaf/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 24, "BATCH_SIZE": 88, "LEARNING_RATE": 1.2e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 4.0, "LABEL_SMOOTHING": 0.10, "EARLY_STOPPING_PATIENCE": 5, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.99, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": False},
    },
    "tomato__fruit": {
        "crop": "tomato", "part": "fruit",
        "ood": "tomato__fruit/ood",
        "oe": "tomato__fruit/oe",
        "oe_enabled": True, "oe_w": 0.25, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 64, "LEARNING_RATE": 6e-5, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 3.0, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.25, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.15},
    },
    "tomato__leaf": {
        "crop": "tomato", "part": "leaf",
        "ood": "tomato__leaf/ood",
        "oe": "tomato__leaf/oe",
        "oe_enabled": True, "oe_w": 0.25, "allow_under_min": False,
        "defaults": {"EPOCHS": 26, "BATCH_SIZE": 88, "LEARNING_RATE": 1.1e-4, "LORA_R": 32, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 3.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.25, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "apricot__fruit": {
        "crop": "apricot", "part": "fruit",
        "ood": "apricot__fruit/ood",
        "oe": "apricot__fruit/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 36, "BATCH_SIZE": 40, "LEARNING_RATE": 1e-4, "LORA_R": 20, "LORA_ALPHA": 20, "LORA_DROPOUT": 0.20, "OOD_FACTOR": 4.5, "LABEL_SMOOTHING": 0.12, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "apricot__leaf": {
        "crop": "apricot", "part": "leaf",
        "ood": "apricot__leaf/ood",
        "oe": "apricot__leaf/oe", "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 38, "BATCH_SIZE": 52, "LEARNING_RATE": 1.2e-4, "LORA_R": 26, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 4.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
})

if ADAPTER_KEY not in ADAPTER_RECS:
    raise ValueError(f"Unsupported ADAPTER_KEY={ADAPTER_KEY!r}. Options: {sorted(ADAPTER_RECS)}")

_adapter_rec = ADAPTER_RECS[ADAPTER_KEY]
CROP_NAME = _adapter_rec["crop"]
PART_NAME = _adapter_rec["part"]
DATASET_NAME = ADAPTER_KEY

# Cell 3, bu parametre hucreden once calistigi icin run kimligini burada dogru adaptere yeniden bagla.
if "ColabLiveTelemetry" in globals() and "TrainingCheckpointManager" in globals():
    RUN_ID = build_notebook_run_id(CROP_NAME, PART_NAME)
    NOTEBOOK_NAME = str(globals().get("NOTEBOOK_NAME", "2_train_continual_sd_lora_adapter.ipynb"))
    NOTEBOOK_FILENAME = str(globals().get("NOTEBOOK_FILENAME", "2_train_continual_sd_lora_adapter.executed.ipynb"))
    REPO_RUN_DIR = build_notebook_run_dir(ROOT, CROP_NAME, PART_NAME, RUN_ID)
    REPO_NOTEBOOK_OUTPUT_PATH = REPO_RUN_DIR / "notebooks" / NOTEBOOK_FILENAME
    LOCAL_OUTPUT_DIR = ROOT / "outputs" / "colab_notebook_training"
    REPO_OUTPUT_DIR = REPO_RUN_DIR / "outputs" / "colab_notebook_training"
    REPO_TELEMETRY_DIR = REPO_RUN_DIR / "telemetry"
    REPO_CHECKPOINT_STATE_DIR = REPO_RUN_DIR / "checkpoint_state"
    LOCAL_TELEMETRY_ROOT = ROOT / "outputs" / "colab_notebook_training" / "telemetry_runtime"
    LOCAL_TELEMETRY_SPOOL_ROOT = ROOT / ".runtime_tmp" / "colab_notebook_training" / "telemetry_spool"
    TELEMETRY = ColabLiveTelemetry(
        notebook_name=NOTEBOOK_NAME,
        run_id=RUN_ID,
        drive_root=LOCAL_TELEMETRY_ROOT,
        local_root=LOCAL_TELEMETRY_SPOOL_ROOT,
    )
    CHECKPOINT_ROOT = TELEMETRY.drive_run_dir
    CHECKPOINT_MANAGER = TrainingCheckpointManager(CHECKPOINT_ROOT, retention=3)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPO_NOTEBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TELEMETRY.configure_repo_output_export(
        notebook_path=REPO_NOTEBOOK_OUTPUT_PATH,
        export_notebook_fn=export_current_colab_notebook,
    )
    TELEMETRY.update_latest({"phase": "adapter_key_selected", "run_id": RUN_ID, "adapter_key": ADAPTER_KEY, "crop": CROP_NAME, "part": PART_NAME})

# --- Notebook 2 parametreleri ---
# Bu hucreyi duzenleyin, sonra kalan hucreleri sirayla calistirin.
# Kosu kimligi icin CROP_NAME/PART_NAME degerlerini ustteki hucreden yonetin.

# Runtime datasets are fetched from one fixed immutable private GitHub Release into this local cache.
DATASET_RELEASE_REPOSITORY = str(
    globals().get("DATASET_RELEASE_REPOSITORY", "EfeErim/bitirmeprojesi")
).strip()
DATASET_RELEASE_TAG = str(globals().get("DATASET_RELEASE_TAG", "aads-dataset-v1.0.0")).strip()
DATASET_RELEASE_CACHE_ROOT = str(
    globals().get("DATASET_RELEASE_CACHE_ROOT", ".runtime_tmp/dataset_release_cache")
).strip()
RUNTIME_DATASET_ROOT = str(globals().get("RUNTIME_DATASET_ROOT", "")).strip()
DATASET_SOURCE_KIND = "github_release"
DATASET_RELEASE_MANIFEST_PATH = str(
    globals().get(
        "DATASET_RELEASE_MANIFEST_PATH",
        "docs/evidence/current/dataset_release/phase9_full_github_release_manifest.json",
    )
).strip()
ALLOW_LOCAL_LEGACY_DATASET = False
DATASET_LEGACY_COMPATIBILITY_REASON = ""

# DATASET_NAME: Notebook 0'un urettigi runtime dataset klasor adi. Bos ise notebook kullaniciya sorar.
DATASET_NAME = ""

# OOD_ROOT: Gercek OOD klasoru. Bos ise ASK_FOR_OOD_ROOT=True iken notebook yol sorar; Enter varsa runtime ood/ kullanir.
OOD_ROOT = ""
ASK_FOR_OOD_ROOT = True

# OE_ROOT: Outlier Exposure egitim klasoru. OE_ENABLED=True ve bos ise ASK_FOR_OE_ROOT=True iken notebook yol sorar; Enter varsa runtime oe/ kullanir.
OE_ROOT = ""
ASK_FOR_OE_ROOT = True
OE_ENABLED = False
OE_LOSS_WEIGHT = 0.5

# CROP_NAME ve PART_NAME, kosu adlandirmasi ve metadata icin kullanilir.
CROP_NAME = globals().get("CROP_NAME", "tomato")
PART_NAME = globals().get("PART_NAME", "unspecified")
ENABLE_BAYESIAN_OPTIMIZATION = bool(globals().get("ENABLE_BAYESIAN_OPTIMIZATION", True))

# ALLOW_UNDER_MIN_TRAINING: True olursa 100 image/class production guardrail'i research kosulari icin bypass edilir.
ALLOW_UNDER_MIN_TRAINING = False

# EPOCHS: train split uzerinden kac tam gecis yapilacagi.
EPOCHS = 12

# BATCH_SIZE: optimizer adimi basina ornek sayisi. GPU limitine yakin olacak sekilde artirilabilir.
BATCH_SIZE = 96

# LEARNING_RATE: adapter/LoRA parametreleri icin optimizer adim buyuklugu.
LEARNING_RATE = 2e-4
# LORA_R: LoRA rank degeri. Buyudukce kapasite ve VRAM/islem maliyeti artar.
LORA_R = 24

# LORA_ALPHA: LoRA olcekleme katsayisi. Genelde LORA_R degerinin 2x-4x araliginda kullanilir.
LORA_ALPHA = 24

# LORA_DROPOUT: LoRA katmanlarina uygulanan dropout. Buyudukce regularizasyon artar.
LORA_DROPOUT = 0.1

# OOD_FACTOR: OOD esik hassasiyetini carpansal olarak ayarlar.
OOD_FACTOR = 3.0
SURE_SEMANTIC_PERCENTILE = 90.0
SURE_CONFIDENCE_PERCENTILE = 97.0
CONFORMAL_ALPHA = 0.05
CONFORMAL_METHOD = "raps"
CONFORMAL_RAPS_LAMBDA = 0.2
CONFORMAL_RAPS_K_REG = 1

# BER_ENABLED: eski/yeni sinif enerji ayrimi icin deneysel egitim regularizeri.
BER_ENABLED = False
# BER_LAMBDA_OLD / BER_LAMBDA_NEW: eski ve yeni sinif kisimlari icin BER ceza agirliklari.
BER_LAMBDA_OLD = 0.1
BER_LAMBDA_NEW = 0.1
BER_WARMUP_STEPS = 50

WEIGHT_DECAY = 0.01

# MIXED_PRECISION: {'off', 'auto', 'fp16', 'bf16'} seceneklerinden biri.
MIXED_PRECISION = "bf16"

GRAD_ACCUM_STEPS = 1

# MAX_GRAD_NORM: gradient clipping esigi. 0 olursa clipping kapanir.
MAX_GRAD_NORM = 1.0

# LABEL_SMOOTHING: CE label smoothing katsayisi.
LABEL_SMOOTHING = 0.0

# LOSS_NAME / LOGITNORM_TAU: varsayilan kayip LogitNorm'dur; CE icin loss_name='cross_entropy' secin.
LOSS_NAME = "logitnorm"
LOGITNORM_TAU = 1.0

# Scheduler ayarlari.
SCHEDULER_NAME = "cosine"
SCHEDULER_WARMUP_RATIO = 0.1
SCHEDULER_MIN_LR = 1e-6

# Early stopping ayarlari.
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0

# Tekrarlanabilirlik ve runtime ayarlari.
DETERMINISTIC = False
TF32_ENABLED = True
SEED = 42

# NUM_WORKERS: dataloader worker sayisi. CPU veri yukleme paralelligini belirler.
NUM_WORKERS = 12

# PREFETCH: NUM_WORKERS > 0 iken worker basina prefetch katsayisi.
PREFETCH = 8

# PIN_MEMORY: host memory sabitleyerek host-to-GPU aktarimini hizlandirir.
PIN_MEMORY = True

# USE_CACHE: destekleniyorsa decode edilmis ornekleri RAM'de tutar.
USE_CACHE = True

# CACHE_TRAIN_SPLIT: continual/train split'ini de cache'ler. Yuksek RAM'li Colab icin uygundur.
CACHE_TRAIN_SPLIT = True

# VALIDATION_EVERY_N_EPOCHS: her N epoch'ta tam validation calistirir; final epoch her zaman dahildir.
VALIDATION_EVERY_N_EPOCHS = 1

# CHECKPOINT_EVERY_N_STEPS / CHECKPOINT_ON_EXCEPTION: notebook checkpoint sikligi ayarlari.
CHECKPOINT_EVERY_N_STEPS = 250
CHECKPOINT_ON_EXCEPTION = True

# STDOUT_BATCH_INTERVAL: canli training ilerleme yazdirma araligi.
STDOUT_BATCH_INTERVAL = 12

# RESUME_MODE: "fresh" yeni kosu baslatir, "resume" son checkpointten devam eder.
RESUME_MODE = "fresh"  # "fresh" or "resume"

# AUTO_DISCONNECT_RUNTIME: tum final exportlar basariliysa Colab runtime'i kapatir.
AUTO_DISCONNECT_RUNTIME = True

# AUTO_DISCONNECT_GRACE_SECONDS: son durum gorunsun diye disconnect oncesi kisa bekleme suresi.
AUTO_DISCONNECT_GRACE_SECONDS = 20

# AUTO_PUSH_TO_GITHUB: final exportlar bitince runs/<crop>/<part>/<RUN_ID>/ klasorunu repoya commit edip pushlar.
AUTO_PUSH_TO_GITHUB = True

# AUTO_PUSH_REMOTE_NAME: auto-push aciksa kullanilacak git remote adi.
AUTO_PUSH_REMOTE_NAME = "origin"

# AUTO_PUSH_BRANCH: auto-push icin branch override degeri. None olursa mevcut branch kullanilir.
AUTO_PUSH_BRANCH = None

# ADAPTER_KEY secimi bu noktada tum notebook degerlerini kesin olarak ezer.
rec = ADAPTER_RECS[ADAPTER_KEY]
CROP_NAME = rec["crop"]
PART_NAME = rec["part"]
DATASET_NAME = ADAPTER_KEY
RUNTIME_DATASET_ROOT = str(globals().get("RUNTIME_DATASET_ROOT", "")).strip()
OOD_ROOT = rec["ood"]
ASK_FOR_OOD_ROOT = False
OE_ROOT = rec["oe"]
ASK_FOR_OE_ROOT = False
OE_ENABLED = bool(rec["oe_enabled"])
OE_LOSS_WEIGHT = float(rec["oe_w"])
ALLOW_UNDER_MIN_TRAINING = bool(rec["allow_under_min"])
VALIDATION_EVERY_N_EPOCHS = 1

_adapter_default_overrides = {
    **rec["defaults"],
    **_USER_DEFAULT_RUNTIME_PARAMS,
}
_adapter_default_overrides.setdefault("WEIGHT_DECAY", 0.01)
_adapter_default_overrides.setdefault("LOSS_NAME", "logitnorm")
_adapter_default_overrides.setdefault("LOGITNORM_TAU", 1.0)
_adapter_default_overrides.setdefault("MIXED_PRECISION", "bf16")
_adapter_default_overrides.setdefault("GRAD_ACCUM_STEPS", 1)
_adapter_default_overrides.setdefault("VALIDATION_EVERY_N_EPOCHS", 1)
_adapter_default_overrides.setdefault("EARLY_STOPPING_PATIENCE", 6)
_adapter_default_overrides.setdefault("RANDAUGMENT_NUM_OPS", 2)
_adapter_default_overrides.setdefault("RANDAUGMENT_MAGNITUDE", 7)
_adapter_default_overrides.setdefault("NUM_WORKERS", 12)
_adapter_default_overrides.setdefault("PREFETCH", 8)
_adapter_default_overrides.setdefault("CACHE_TRAIN_SPLIT", True)
_adapter_default_overrides["ENABLE_BAYESIAN_OPTIMIZATION"] = bool(
    _adapter_default_overrides.get("ENABLE_BAYESIAN_OPTIMIZATION", ENABLE_BAYESIAN_OPTIMIZATION)
)
ENABLE_BAYESIAN_OPTIMIZATION = bool(_adapter_default_overrides["ENABLE_BAYESIAN_OPTIMIZATION"])
MANUAL_PARAM_OVERRIDES = {}
MANUAL_PARAM_OVERRIDES = {
    **_adapter_default_overrides,
    **_USER_MANUAL_PARAM_OVERRIDES,
}
print(
    f"[ADAPTER_SELECTED] key={ADAPTER_KEY} crop={CROP_NAME} part={PART_NAME} "
    f"dataset={DATASET_NAME} ood={OOD_ROOT} oe_enabled={OE_ENABLED} oe={OE_ROOT} "
    f"run_id={RUN_ID}"
)
print(
    f"[TRAINING_PLAN] epochs={MANUAL_PARAM_OVERRIDES['EPOCHS']} batch={MANUAL_PARAM_OVERRIDES['BATCH_SIZE']} "
    f"lr={MANUAL_PARAM_OVERRIDES['LEARNING_RATE']} lora_r={MANUAL_PARAM_OVERRIDES['LORA_R']} "
    f"allow_under_min={ALLOW_UNDER_MIN_TRAINING} validation_every={VALIDATION_EVERY_N_EPOCHS}"
)
if _USER_MANUAL_PARAM_OVERRIDES:
    print(f"[MANUAL_OVERRIDES] applied={sorted(_USER_MANUAL_PARAM_OVERRIDES)}")
print("[SONRAKI] Erisim kontrolu -> dataset validation -> engine init -> training -> calibration -> save/eval hucresini sirayla calistirin.")


# Notebook icinde daha sonra kullanilacak gizli repo varsayimlarini yukle.
BASE_CONFIG = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
CONTINUAL_DATA_CFG = BASE_CONFIG.get("training", {}).get("continual", {}).get("data", {})

# Ensure torch is imported (lazy-loaded from runtime_setup)
_ensure_torch()

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = bool(TF32_ENABLED)

TARGET_SIZE = int(CONTINUAL_DATA_CFG.get("target_size", 224))
DATA_SAMPLER = str(CONTINUAL_DATA_CFG.get("sampler", "shuffle"))
AUGMENTATION_POLICY = str(CONTINUAL_DATA_CFG.get("augmentation_policy", "randaugment")).strip().lower()
RANDAUGMENT_NUM_OPS = int(CONTINUAL_DATA_CFG.get("randaugment_num_ops", 2))
RANDAUGMENT_MAGNITUDE = int(CONTINUAL_DATA_CFG.get("randaugment_magnitude", 7))
ALLOW_UNDER_MIN_TRAINING = bool(ALLOW_UNDER_MIN_TRAINING)
LOADER_ERROR_POLICY = str(CONTINUAL_DATA_CFG.get("loader_error_policy", "tolerant"))
CACHE_SIZE = int(CONTINUAL_DATA_CFG.get("cache_size", 1000))
VALIDATE_IMAGES_ON_INIT = bool(CONTINUAL_DATA_CFG.get("validate_images_on_init", True))

from scripts.colab_training_recommendations import resolve_notebook_params

def _collect_notebook_base_params():
    return {
        "OE_ENABLED": bool(OE_ENABLED),
        "ALLOW_UNDER_MIN_TRAINING": bool(ALLOW_UNDER_MIN_TRAINING),
        "EPOCHS": int(EPOCHS),
        "BATCH_SIZE": int(BATCH_SIZE),
        "LEARNING_RATE": float(LEARNING_RATE),
        "LORA_R": int(LORA_R),
        "LORA_ALPHA": int(LORA_ALPHA),
        "LORA_DROPOUT": float(LORA_DROPOUT),
        "OOD_FACTOR": float(OOD_FACTOR),
        "SURE_SEMANTIC_PERCENTILE": float(SURE_SEMANTIC_PERCENTILE),
        "SURE_CONFIDENCE_PERCENTILE": float(SURE_CONFIDENCE_PERCENTILE),
        "CONFORMAL_ALPHA": float(CONFORMAL_ALPHA),
        "CONFORMAL_METHOD": str(CONFORMAL_METHOD),
        "CONFORMAL_RAPS_LAMBDA": float(CONFORMAL_RAPS_LAMBDA),
        "CONFORMAL_RAPS_K_REG": int(CONFORMAL_RAPS_K_REG),
        "BER_ENABLED": bool(BER_ENABLED),
        "BER_LAMBDA_OLD": float(BER_LAMBDA_OLD),
        "BER_LAMBDA_NEW": float(BER_LAMBDA_NEW),
        "BER_WARMUP_STEPS": int(BER_WARMUP_STEPS),
        "RANDAUGMENT_NUM_OPS": int(RANDAUGMENT_NUM_OPS),
        "RANDAUGMENT_MAGNITUDE": int(RANDAUGMENT_MAGNITUDE),
        "WEIGHT_DECAY": float(WEIGHT_DECAY),
        "MIXED_PRECISION": str(MIXED_PRECISION),
        "GRAD_ACCUM_STEPS": int(GRAD_ACCUM_STEPS),
        "MAX_GRAD_NORM": float(MAX_GRAD_NORM),
        "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
        "LOSS_NAME": str(LOSS_NAME),
        "LOGITNORM_TAU": float(LOGITNORM_TAU),
        "SCHEDULER_NAME": str(SCHEDULER_NAME),
        "SCHEDULER_WARMUP_RATIO": float(SCHEDULER_WARMUP_RATIO),
        "SCHEDULER_MIN_LR": float(SCHEDULER_MIN_LR),
        "EARLY_STOPPING_PATIENCE": int(EARLY_STOPPING_PATIENCE),
        "EARLY_STOPPING_MIN_DELTA": float(EARLY_STOPPING_MIN_DELTA),
        "DETERMINISTIC": bool(DETERMINISTIC),
        "SEED": int(SEED),
        "NUM_WORKERS": int(NUM_WORKERS),
        "PREFETCH": int(PREFETCH),
        "PIN_MEMORY": bool(PIN_MEMORY),
        "USE_CACHE": bool(USE_CACHE),
        "CACHE_TRAIN_SPLIT": bool(CACHE_TRAIN_SPLIT),
        "VALIDATION_EVERY_N_EPOCHS": int(VALIDATION_EVERY_N_EPOCHS),
        "CHECKPOINT_EVERY_N_STEPS": int(CHECKPOINT_EVERY_N_STEPS),
        "CHECKPOINT_ON_EXCEPTION": bool(CHECKPOINT_ON_EXCEPTION),
        "STDOUT_BATCH_INTERVAL": int(STDOUT_BATCH_INTERVAL),
        "RESUME_MODE": str(RESUME_MODE),
        "AUTO_DISCONNECT_RUNTIME": bool(AUTO_DISCONNECT_RUNTIME),
        "AUTO_DISCONNECT_GRACE_SECONDS": int(AUTO_DISCONNECT_GRACE_SECONDS),
        "AUTO_PUSH_TO_GITHUB": bool(AUTO_PUSH_TO_GITHUB),
        "AUTO_PUSH_REMOTE_NAME": str(AUTO_PUSH_REMOTE_NAME),
        "AUTO_PUSH_BRANCH": AUTO_PUSH_BRANCH,
        "ENABLE_BAYESIAN_OPTIMIZATION": bool(ENABLE_BAYESIAN_OPTIMIZATION),
    }

resolved_manual_overrides = dict(MANUAL_PARAM_OVERRIDES or {})

INITIAL_EFFECTIVE_PARAMS = resolve_notebook_params(
    _collect_notebook_base_params(),
    resolved_manual_overrides,
)
OE_ENABLED = bool(INITIAL_EFFECTIVE_PARAMS["OE_ENABLED"])

STATE = {
    "validated": False,
    "class_names": [],
    "runtime_dataset_root": None,
    "runtime_dataset_key": None,
    "selected_dataset_name": None,
    "selected_dataset_root": None,
    "resolved_ood_root": None,
    "resolved_oe_root": None,
    "dataset_inspection": {},
    "hardware_inspection": {},
    "recommendation_report": {},
    "recommendation_decision": "disabled",
    "effective_params": dict(INITIAL_EFFECTIVE_PARAMS),
    "adapter": None,
    "loaders": None,
    "history": None,
    "calibration": None,
    "checkpoint_manager": CHECKPOINT_MANAGER,
    "resume_manifest": None,
    "best_val_loss": None,
    "best_metric_state": {},
    "auto_disconnect_report": None,
    "git_push_report": None,
    "telemetry": TELEMETRY,
}

if str(INITIAL_EFFECTIVE_PARAMS.get("RESUME_MODE", RESUME_MODE)) == "resume":
    try:
        STATE["resume_manifest"] = CHECKPOINT_MANAGER.get_latest()
        if STATE["resume_manifest"]:
            manifest = STATE["resume_manifest"]
            print(
                f"[RESUME] Checkpoint bulundu: {manifest.get('name', '?')} "
                f"epoch={manifest.get('epoch', 0)} step={manifest.get('global_step', 0)}"
            )
    except Exception:
        pass

hf_token_ready = False
hf_token = str(resolve_hf_token() or "").strip()
if not hf_token:
    raise RuntimeError(
        "Notebook 2 egitim baslamadan once Hugging Face token ister. "
        "HF_TOKEN, HUGGINGFACE_TOKEN veya HUGGINGFACE_HUB_TOKEN degerini env var ya da Colab secret olarak tanimlayin."
    )
hf_token_ready = bool(login_and_check_hf_token(print_fn=print))
if not hf_token_ready:
    raise RuntimeError(
        "Hugging Face token on kontrolu basarisiz oldu. Egitimden once tokeni duzeltin."
    )

github_token_ready = False
if bool(INITIAL_EFFECTIVE_PARAMS.get("AUTO_PUSH_TO_GITHUB", AUTO_PUSH_TO_GITHUB)):
    github_token = str(resolve_github_token() or "").strip()
    if not github_token:
        raise RuntimeError(
            "AUTO_PUSH_TO_GITHUB is enabled, but no GitHub token was found. "
            "Egitim baslamadan once GH_TOKEN veya GITHUB_TOKEN degerini env var ya da Colab secret olarak tanimlayin."
        )
    github_token_ready = True
    print("[GIT] Auto-push on kontrolu gecti: GitHub token cozuldu.")

print(
    f"[CONFIG] source=notebook_cell crop={CROP_NAME} epochs={EPOCHS} bs={BATCH_SIZE} "
    f"lr={LEARNING_RATE} resume={RESUME_MODE} ber={BER_ENABLED} "
    f"loss={LOSS_NAME} tau={LOGITNORM_TAU} auto_disconnect={AUTO_DISCONNECT_RUNTIME} "
    f"auto_push={AUTO_PUSH_TO_GITHUB} bayes_opt={ENABLE_BAYESIAN_OPTIMIZATION}"
)
print(
    f"[DATASET] release={DATASET_RELEASE_REPOSITORY}@{DATASET_RELEASE_TAG} "
    f"cache={DATASET_RELEASE_CACHE_ROOT} dataset_name={DATASET_NAME or ADAPTER_KEY}"
)
print(
    f"[DATASET][LINEAGE] source_kind={DATASET_SOURCE_KIND} "
    f"release_manifest={DATASET_RELEASE_MANIFEST_PATH or '<none>'}"
)
print(
    f"[OOD] ood_root={OOD_ROOT or '<ask>'} ask_for_ood_root={ASK_FOR_OOD_ROOT}"
)
print(
    f"[OE] oe_root={OE_ROOT or '<ask>'} ask_for_oe_root={ASK_FOR_OE_ROOT} "
    f"enabled={OE_ENABLED} loss_weight={OE_LOSS_WEIGHT}"
)
print(
    f"[RUNTIME] defaults=notebook_cell mp={MIXED_PRECISION} workers={NUM_WORKERS} prefetch={PREFETCH} "
    f"sched={SCHEDULER_NAME} wd={WEIGHT_DECAY} accum={GRAD_ACCUM_STEPS} grad_clip={MAX_GRAD_NORM} "
    f"label_smooth={LABEL_SMOOTHING} warmup={SCHEDULER_WARMUP_RATIO} "
    f"early_stop={EARLY_STOPPING_PATIENCE}/{EARLY_STOPPING_MIN_DELTA} "
    f"val_every={VALIDATION_EVERY_N_EPOCHS} cache_train={CACHE_TRAIN_SPLIT} "
    f"aug={AUGMENTATION_POLICY} randaug={RANDAUGMENT_NUM_OPS}/{RANDAUGMENT_MAGNITUDE} "
)
print(
    f"[OOD] factor={OOD_FACTOR} sure={SURE_SEMANTIC_PERCENTILE}/{SURE_CONFIDENCE_PERCENTILE} "
    f"conformal={CONFORMAL_METHOD} alpha={CONFORMAL_ALPHA} "
    f"raps_lambda={CONFORMAL_RAPS_LAMBDA} raps_k={CONFORMAL_RAPS_K_REG}"
)
TELEMETRY.update_latest(
    {
        "phase": "parameters_ready",
        "parameter_source": "notebook_cell",
        "crop": CROP_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_name": DATASET_NAME,
        "dataset_source_kind": DATASET_SOURCE_KIND,
        "dataset_release_repository": DATASET_RELEASE_REPOSITORY,
        "dataset_release_tag": DATASET_RELEASE_TAG,
        "dataset_release_cache_root": DATASET_RELEASE_CACHE_ROOT,
        "dataset_release_manifest_path": DATASET_RELEASE_MANIFEST_PATH,
        "ood_root": OOD_ROOT,
        "ask_for_ood_root": ASK_FOR_OOD_ROOT,
        "oe_root": OE_ROOT,
        "ask_for_oe_root": ASK_FOR_OE_ROOT,
        "oe_enabled": OE_ENABLED,
        "oe_loss_weight": OE_LOSS_WEIGHT,
        "loss_name": LOSS_NAME,
        "logitnorm_tau": LOGITNORM_TAU,
        "mixed_precision": MIXED_PRECISION,
        "num_workers": NUM_WORKERS,
        "prefetch": PREFETCH,
        "cache_train_split": CACHE_TRAIN_SPLIT,
        "augmentation_policy": AUGMENTATION_POLICY,
        "randaugment_num_ops": RANDAUGMENT_NUM_OPS,
        "randaugment_magnitude": RANDAUGMENT_MAGNITUDE,
        "allow_under_min_training": ALLOW_UNDER_MIN_TRAINING,
        "ber_enabled": BER_ENABLED,
        "ber_lambda_old": BER_LAMBDA_OLD,
        "ber_lambda_new": BER_LAMBDA_NEW,
        "ber_warmup_steps": BER_WARMUP_STEPS,
        "ood_factor": OOD_FACTOR,
        "sure_semantic_percentile": SURE_SEMANTIC_PERCENTILE,
        "sure_confidence_percentile": SURE_CONFIDENCE_PERCENTILE,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_method": CONFORMAL_METHOD,
        "conformal_raps_lambda": CONFORMAL_RAPS_LAMBDA,
        "conformal_raps_k_reg": CONFORMAL_RAPS_K_REG,
        "label_smoothing": LABEL_SMOOTHING,
        "scheduler_warmup_ratio": SCHEDULER_WARMUP_RATIO,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "resume_mode": RESUME_MODE,
        "validation_every_n_epochs": VALIDATION_EVERY_N_EPOCHS,
        "enable_bayesian_optimization": bool(ENABLE_BAYESIAN_OPTIMIZATION),
        "auto_push_to_github": AUTO_PUSH_TO_GITHUB,
        "hf_token_ready": hf_token_ready,
        "auto_push_token_ready": github_token_ready,
        "auto_push_remote_name": AUTO_PUSH_REMOTE_NAME,
        "auto_push_branch": AUTO_PUSH_BRANCH,
        "auto_disconnect_runtime": AUTO_DISCONNECT_RUNTIME,
        "auto_disconnect_grace_seconds": AUTO_DISCONNECT_GRACE_SECONDS,
    }
)
