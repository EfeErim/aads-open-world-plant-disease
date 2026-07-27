from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from scripts.colab_live_telemetry import ColabLiveTelemetry
from scripts.colab_repo_bootstrap import (
    export_current_colab_notebook,
    mirror_checkpoint_state_to_repo,
    mirror_path_to_repo,
)
from scripts.prepare_grouped_runtime_dataset import resolve_safe_embedding_batch_size
from src.core.config_manager import ConfigurationManager
from src.data.dataset_release_runtime import (
    DatasetReleaseAccessBlocker as DatasetReleaseAccessBlocker,
)
from src.data.dataset_release_runtime import (
    fetch_materialize_dataset_release,
    resolve_dataset_release_manifest,
)


def run_bootstrap_notebook(notebook_name: str = "Notebook 0: Dataset Preparation", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status

    BOOTSTRAP = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    print_bootstrap_status(BOOTSTRAP)
    return BOOTSTRAP


def init_run_environment(ROOT: Path) -> dict:
    STATE: dict = {}
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    NOTEBOOK_FILENAME = '0_grouped_dataset_preparation.executed.ipynb'
    REPO_RUN_DIR = ROOT / 'runs' / RUN_ID
    REPO_NOTEBOOK_OUTPUT_PATH = REPO_RUN_DIR / 'notebooks' / NOTEBOOK_FILENAME
    LOCAL_OUTPUT_DIR = ROOT / 'outputs' / 'colab_notebook_data_prep'
    REPO_OUTPUT_DIR = REPO_RUN_DIR / 'outputs' / 'colab_notebook_data_prep'
    REPO_TELEMETRY_DIR = REPO_RUN_DIR / 'telemetry'
    REPO_CHECKPOINT_STATE_DIR = REPO_RUN_DIR / 'checkpoint_state'

    TELEMETRY = ColabLiveTelemetry(
        notebook_name='0_grouped_dataset_preparation.ipynb',
        run_id=RUN_ID,
    )
    CHECKPOINT_ROOT = TELEMETRY.drive_run_dir if TELEMETRY.drive_run_dir.exists() else TELEMETRY.local_run_dir

    REQUESTED_DEVICE = str(ConfigurationManager(config_dir=str(ROOT / 'config'), environment='colab').load_all_configs().get('training', {}).get('continual', {}).get('device', 'cuda'))
    DEVICE = REQUESTED_DEVICE if not REQUESTED_DEVICE.startswith('cuda') or torch.cuda.is_available() else 'cpu'

    EMBEDDING_BATCH_SIZE = resolve_safe_embedding_batch_size(DEVICE)

    # System RAM optimization for low-memory Colab environments
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9
        total_ram_gb = psutil.virtual_memory().total / 1e9
        # Low memory mode: if available RAM < 8GB or total < 15GB
        LOW_MEMORY_MODE = available_ram_gb < 8 or total_ram_gb < 15
        MAX_IMAGES_IN_MEMORY = 256 if not LOW_MEMORY_MODE else 64
    except Exception:
        LOW_MEMORY_MODE = True
        MAX_IMAGES_IN_MEMORY = 64

    NEIGHBORS = 4

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPO_NOTEBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    TELEMETRY.configure_repo_output_export(
        output_dir=REPO_OUTPUT_DIR,
        notebook_filename=NOTEBOOK_FILENAME,
        export_notebook_fn=export_current_colab_notebook,
    )
    TELEMETRY.update_latest({'phase': 'bootstrap_ready', 'run_id': RUN_ID})

    return {
        'STATE': STATE,
        'RUN_ID': RUN_ID,
        'NOTEBOOK_FILENAME': NOTEBOOK_FILENAME,
        'REPO_RUN_DIR': REPO_RUN_DIR,
        'REPO_NOTEBOOK_OUTPUT_PATH': REPO_NOTEBOOK_OUTPUT_PATH,
        'LOCAL_OUTPUT_DIR': LOCAL_OUTPUT_DIR,
        'REPO_OUTPUT_DIR': REPO_OUTPUT_DIR,
        'REPO_TELEMETRY_DIR': REPO_TELEMETRY_DIR,
        'REPO_CHECKPOINT_STATE_DIR': REPO_CHECKPOINT_STATE_DIR,
        'CHECKPOINT_ROOT': CHECKPOINT_ROOT,
        'TELEMETRY': TELEMETRY,
        'REQUESTED_DEVICE': REQUESTED_DEVICE,
        'DEVICE': DEVICE,
        'EMBEDDING_BATCH_SIZE': EMBEDDING_BATCH_SIZE,
        'LOW_MEMORY_MODE': LOW_MEMORY_MODE,
        'MAX_IMAGES_IN_MEMORY': MAX_IMAGES_IN_MEMORY,
        'NEIGHBORS': NEIGHBORS,
    }


def save_run_outputs_to_repo(LOCAL_OUTPUT_DIR: Path, REPO_OUTPUT_DIR: Path, TELEMETRY: ColabLiveTelemetry, CHECKPOINT_ROOT: Path, REPO_TELEMETRY_DIR: Path, REPO_CHECKPOINT_STATE_DIR: Path) -> dict:
    exports: dict[str, str] = {}
    mirrored_outputs = mirror_path_to_repo(LOCAL_OUTPUT_DIR, REPO_OUTPUT_DIR)
    if mirrored_outputs is not None:
        exports['outputs'] = str(mirrored_outputs)
    telemetry_source = getattr(TELEMETRY, "drive_run_dir", None)
    if telemetry_source is None or not Path(telemetry_source).exists():
        telemetry_source = getattr(TELEMETRY, "local_run_dir", CHECKPOINT_ROOT)
    mirrored_telemetry = mirror_path_to_repo(telemetry_source, REPO_TELEMETRY_DIR)
    if mirrored_telemetry is not None:
        exports['telemetry'] = str(mirrored_telemetry)
    mirrored_checkpoint_state = mirror_checkpoint_state_to_repo(CHECKPOINT_ROOT, REPO_CHECKPOINT_STATE_DIR)
    if mirrored_checkpoint_state is not None:
        exports['checkpoint_state'] = str(mirrored_checkpoint_state)
    return exports


def import_dataset_from_drive(source_path: str | Path, destination_path: str | Path, dataset_name: str, overwrite: bool = False) -> bool:
    source_path = Path(source_path)
    destination_path = Path(destination_path).expanduser()
    destination_path.mkdir(parents=True, exist_ok=True)
    target = destination_path / dataset_name
    import shutil
    if target.exists():
        if not overwrite:
            print(f"[DRIVE] Hedef klasor zaten var: {target}")
            return False
        shutil.rmtree(target)
        print(f"[DRIVE] Mevcut klasor silindi: {target}")
    try:
        print(f"[DRIVE] Kopyalama baslatildi: {source_path} -> {target}")
        shutil.copytree(source_path, target, dirs_exist_ok=False)
        print(f"[DRIVE] Basariyla kopyalandi: {target}")
        return True
    except Exception as e:
        print(f"[DRIVE] Kopyalama basarisiz: {e}")
        return False


def _drive_destination_parent(DATASET_ROOT: str | Path, ROOT: Path) -> Path:
    explicit_root = str(DATASET_ROOT).strip()
    if explicit_root:
        root_path = Path(explicit_root).expanduser()
        return root_path.resolve() if root_path.is_absolute() else (ROOT / root_path).resolve()
    return ROOT / "data" / "imported_from_drive"


def _prompt_yes_no(prompt: str, default_value: bool = True) -> bool:
    suffix = "[Y/n]" if default_value else "[y/N]"
    try:
        raw = str(input(f"{prompt} {suffix} ")).strip().lower()
    except EOFError:
        raw = ""
    if not raw:
        return bool(default_value)
    return raw in {"y", "yes", "evet", "e"}


def run_dataset_audit(ROOT: Path, STATE: dict, TELEMETRY: ColabLiveTelemetry, DEVICE: str, EMBEDDING_BATCH_SIZE: int, NEIGHBORS: int, REPO_DATASET_ROOT: str, REPO_DATASET_NAME: str, DATASET_ROOT: str, CROP_NAME: str, PART_NAME: str, PREP_ARTIFACT_ROOT: str, PREP_DINOV3_MODEL_ID: str, PREP_BIOCLIP_MODEL_ID: str, UNDER_MIN_EVAL_POLICY: dict, IMPORT_FROM_DRIVE: bool = False, DRIVE_DATASET_PATH: str = "", INTERACTIVE_AUDIT_REVIEW: bool = True, MAX_INTERACTIVE_REVIEW_ITEMS: int = 25) -> tuple[dict, str, str]:
    """Run dataset audit and return STATE updates and resolved values."""
    from scripts.colab_dataset_layout import resolve_repo_dataset_directory
    from scripts.evaluate_dataset_layout import evaluate_layout
    from scripts.prepare_grouped_runtime_dataset import (
        build_grouped_dataset_plan,
        build_human_review_packet,
        format_human_review_packet,
    )
    from src.shared.json_utils import write_json

    def _resolve_repo_dataset_root(repo_relative_root: str) -> Path:
        raw_repo_relative_root = str(repo_relative_root or "").strip()
        if not raw_repo_relative_root:
            raise RuntimeError("REPO_DATASET_ROOT bos birakilamaz.")
        candidate = Path(raw_repo_relative_root).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (ROOT / candidate).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError as exc:
                raise RuntimeError(f"REPO_DATASET_ROOT repo disina cikamaz: {raw_repo_relative_root}") from exc
        return resolved

    def _normalize_token(value: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_")

    def _infer_crop_and_part_from_dataset_name(dataset_name: str) -> tuple[str, str]:
        tokens = [token for token in _normalize_token(dataset_name).split("_") if token]
        if not tokens:
            return "", "unspecified"
        if len(tokens) == 1:
            return tokens[0], "unspecified"
        return tokens[0], "_".join(tokens[1:]) or "unspecified"

    def _prompt_text(prompt: str, default_value: str = "") -> str:
        try:
            raw = str(input(prompt)).strip()
        except EOFError:
            raw = ""
        return raw or str(default_value or "").strip()

    explicit_dataset_root = str(DATASET_ROOT).strip()
    if explicit_dataset_root:
        dataset_root = _resolve_repo_dataset_root(explicit_dataset_root)
        dataset_name = dataset_root.name
    else:
        dataset_name, dataset_root, available_dataset_names = resolve_repo_dataset_directory(
            repo_root=ROOT,
            repo_relative_root=REPO_DATASET_ROOT,
            requested_name=REPO_DATASET_NAME,
            prompt_label="class-root dataset",
        )
        print(f"[PREP] repo dataset options={available_dataset_names}")

    dataset_source = "drive" if IMPORT_FROM_DRIVE and str(DRIVE_DATASET_PATH).strip() and str(DATASET_ROOT).strip() else "repo"
    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root bulunamadi: {dataset_root}")

    layout_report = evaluate_layout(root=dataset_root)
    print(
        f"[DATASET] layout_ok={layout_report.get('ok')} "
        f"classes={len(layout_report.get('classes', []))} "
        f"warnings={len(layout_report.get('warnings', []))} "
        f"errors={len(layout_report.get('errors', []))}"
    )

    if not layout_report.get("ok"):
        raise RuntimeError("Dataset layout hatali: " + "; ".join(layout_report.get("errors", [])))

    inferred_crop_name, inferred_part_name = _infer_crop_and_part_from_dataset_name(dataset_name)
    resolved_crop_name = str(CROP_NAME).strip()
    if not resolved_crop_name:
        resolved_crop_name = _prompt_text(f"CROP_NAME bos. '{dataset_name}' dataseti icin crop anahtarini girin [{inferred_crop_name or 'crop'}]: ", inferred_crop_name or "crop")

    resolved_part_name = str(PART_NAME).strip()
    if not resolved_part_name:
        resolved_part_name = _prompt_text(f"PART_NAME bos. '{dataset_name}' dataseti icin part adini girin [{inferred_part_name or 'unspecified'}]: ", inferred_part_name or "unspecified") or "unspecified"
    CROP_NAME = resolved_crop_name
    PART_NAME = resolved_part_name
    print(f"[PREP] secilen crop={CROP_NAME} part={PART_NAME}")

    artifact_root = Path(PREP_ARTIFACT_ROOT).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = (ROOT / artifact_root).resolve()

    summary = build_grouped_dataset_plan(
        class_root=dataset_root,
        crop_name=CROP_NAME,
        artifact_root=artifact_root,
        taxonomy_path=ROOT / "config" / "plant_taxonomy.json",
        dino_model_id=PREP_DINOV3_MODEL_ID,
        bioclip_model_id=PREP_BIOCLIP_MODEL_ID,
        device=DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
        neighbors=NEIGHBORS,
        under_min_eval_policy=UNDER_MIN_EVAL_POLICY,
        progress_fn=lambda message: print(f"[PREP] {message}"),
    )

    STATE["validated"] = True
    STATE["audit_summary"] = summary
    STATE["artifact_root"] = artifact_root
    STATE["dataset_root"] = dataset_root
    STATE["dataset_name"] = dataset_name
    STATE["dataset_source"] = dataset_source
    STATE["crop_name"] = CROP_NAME
    STATE["part_name"] = PART_NAME

    print(f"[PREP] dataset_source={dataset_source} dataset_name={dataset_name} dataset_root={dataset_root}")
    print(f"[PREP] crop_name={CROP_NAME} part_name={PART_NAME}")
    print(json.dumps(summary.get("summary", {}), indent=2))
    print(f"[PREP] runtime_ready={summary.get('runtime_ready')} artifact_root={artifact_root}")
    review_packet = build_human_review_packet(
        summary,
        artifact_root=artifact_root,
        max_review_items=MAX_INTERACTIVE_REVIEW_ITEMS,
    )
    STATE["human_review_packet"] = review_packet
    STATE["human_review_approved"] = not bool(review_packet.get("pause_recommended"))
    write_json(artifact_root / "human_review_packet.json", review_packet, ensure_ascii=False)
    print(format_human_review_packet(review_packet))

    if summary.get("blocking_issues"):
        print("[PREP] Bloklayici sorunlar:")
        for item in summary["blocking_issues"]:
            print(f"  - {item}")
    if INTERACTIVE_AUDIT_REVIEW and review_packet.get("pause_recommended"):
        print("[PREP] Human review gate: artifactleri inceleyebilirsiniz; Enter guvenli varsayilani uygular.")
        if not _prompt_yes_no("Audit gate kararini kabul edip guvenli varsayilan routing ile devam edilsin mi?", True):
            STATE["human_review_stop_requested"] = True
            STATE["human_review_approved"] = False
            raise RuntimeError("Human review tarafindan durduruldu. Artifactleri inceleyip hucreyi yeniden calistirin.")
        STATE["human_review_stop_requested"] = False
        STATE["human_review_approved"] = True
    else:
        STATE["human_review_stop_requested"] = False
        STATE["human_review_approved"] = True

    TELEMETRY.update_latest({
        "phase": "data_prep_audited",
        "dataset_root": str(dataset_root),
        "dataset_name": str(dataset_name),
        "dataset_source": dataset_source,
        "artifact_root": str(artifact_root),
        "runtime_ready": bool(summary.get("runtime_ready")),
    })

    return STATE, resolved_crop_name, resolved_part_name


def run_prepare_dataset(ROOT: Path, STATE: dict, TELEMETRY: ColabLiveTelemetry, DEVICE: str, EMBEDDING_BATCH_SIZE: int, NEIGHBORS: int, CROP_NAME: str, PART_NAME: str, PREPARED_CLASS_ROOT: str, CLEANUP_SEED: int, PREP_DINOV3_MODEL_ID: str, PREP_BIOCLIP_MODEL_ID: str, UNDER_MIN_EVAL_POLICY: dict, PREPARE_DATASET_FROM_REPORTS: bool) -> None:
    """Prepare dataset from audit reports."""
    from scripts.prepare_grouped_runtime_dataset import build_prepared_dataset_key
    from scripts.prepare_materialization_dataset import prepare_class_root_for_materialization

    if not STATE.get("validated") or STATE.get("audit_summary") is None:
        raise RuntimeError("Once dataset audit hucresini calistirin.")

    summary = STATE["audit_summary"]
    if not PREPARE_DATASET_FROM_REPORTS:
        print("[PREP] PREPARE_DATASET_FROM_REPORTS=False. Audit raporlari pasif birakildi.")
    else:
        dataset_key = build_prepared_dataset_key(CROP_NAME, PART_NAME)
        prepared_class_root_parent = Path(PREPARED_CLASS_ROOT).expanduser()
        if not prepared_class_root_parent.is_absolute():
            prepared_class_root_parent = (ROOT / prepared_class_root_parent).resolve()
        prepared_class_root = prepared_class_root_parent / dataset_key
        prepared_artifact_root = STATE["artifact_root"].parent / f"{STATE['artifact_root'].name}_prepared"
        prep_counts = dict(summary.get("summary", {}))

        print("[PREP] Rapor tabanli hazirlik ozeti:")
        print(f"  dataset_key={dataset_key} total_images={prep_counts.get('total_images', 0)} cross_class_conflicts={prep_counts.get('cross_class_conflicts', 0)} same_class_high_risk_clusters={prep_counts.get('same_class_high_risk_clusters', 0)}")
        print(f"  source_dataset_root={STATE['dataset_root']}")
        print(f"  source_artifact_root={STATE['artifact_root']}")
        print(f"  prepared_class_root={prepared_class_root}")
        print(f"  prepared_artifact_root={prepared_artifact_root}")
        print(f"  cleanup_seed={CLEANUP_SEED}")

        prep_result = prepare_class_root_for_materialization(
            class_root=STATE["dataset_root"],
            crop_name=CROP_NAME,
            part_name=PART_NAME,
            audit_artifact_root=STATE["artifact_root"],
            prepared_class_root=prepared_class_root,
            prepared_artifact_root=prepared_artifact_root,
            taxonomy_path=ROOT / "config" / "plant_taxonomy.json",
            dino_model_id=PREP_DINOV3_MODEL_ID,
            bioclip_model_id=PREP_BIOCLIP_MODEL_ID,
            device=DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
            neighbors=NEIGHBORS,
            cleanup_seed=CLEANUP_SEED,
            quarantine_cross_class_conflicts=True,
            under_min_eval_policy=UNDER_MIN_EVAL_POLICY,
            materialization_strategy="auto",
            progress_fn=lambda message: print(f"[PREP] {message}"),
        )

        STATE["prep_materialization_result"] = prep_result
        STATE["dataset_root"] = Path(prep_result["prepared_class_root"])
        STATE["dataset_source"] = "prepared_class_root"
        STATE["artifact_root"] = Path(prep_result["prepared_artifact_root"])
        STATE["audit_summary"] = prep_result["rerun_summary"]
        summary = STATE["audit_summary"]

        print(f"[PREP] prepared_runtime_ready={prep_result.get('prepared_runtime_ready')} dataset_key={prep_result.get('dataset_key')} prepared_class_root={STATE['dataset_root']}")
        print(f"[PREP] Hazirlik sonrasi artifact_root={STATE['artifact_root']}")

        TELEMETRY.update_latest({
            "phase": "data_prep_prepared",
            "dataset_root": str(STATE["dataset_root"]),
            "dataset_source": str(STATE.get("dataset_source") or "prepared_class_root"),
            "artifact_root": str(STATE["artifact_root"]),
            "runtime_ready": bool(summary.get("runtime_ready")),
        })


def run_materialize_runtime_dataset(
    ROOT: Path,
    STATE: dict,
    TELEMETRY: ColabLiveTelemetry,
    CROP_NAME: str,
    PART_NAME: str,
    OOD_ROOT: str,
    OOD_DATASET_NAME: str,
    OOD_DATASET_ROOT: str,
    ASK_FOR_OOD_ROOT: bool,
    PREPARED_RUNTIME_ROOT: str,
    MATERIALIZE_AFTER_REVIEW: bool,
    REPO_NOTEBOOK_OUTPUT_PATH: Path,
    REPO_RUN_DIR: Path,
    REPO_RUN_EXPORTS: dict,
    DATASET_RELEASE_REPOSITORY: str = "",
    DATASET_RELEASE_TAG: str = "",
    DATASET_RELEASE_TARGET: str = "",
    DATASET_RELEASE_CACHE_ROOT: str = ".runtime_tmp/dataset_release_cache",
    INTERACTIVE_AUDIT_REVIEW: bool = True,
) -> None:
    from scripts.colab_dataset_layout import resolve_repo_dataset_directory
    from scripts.prepare_grouped_runtime_dataset import (
        build_prepared_dataset_key,
        format_human_review_packet,
        materialize_grouped_runtime_dataset,
    )

    if not STATE.get("validated") or STATE.get("audit_summary") is None:
        raise RuntimeError("Once dataset audit hucresini calistirin.")

    def _resolve_optional_ood_root() -> Path | None:
        explicit_ood_root = str(OOD_ROOT).strip()
        requested_ood_dataset = str(OOD_DATASET_NAME).strip()
        if explicit_ood_root and requested_ood_dataset:
            raise RuntimeError("OOD_ROOT ve OOD_DATASET_NAME ayni anda kullanilamaz.")
        if explicit_ood_root:
            resolved_ood_root = Path(explicit_ood_root).expanduser()
            if not resolved_ood_root.is_absolute():
                resolved_ood_root = (ROOT / resolved_ood_root).resolve()
            if not resolved_ood_root.is_dir():
                raise RuntimeError(f"OOD root not found or not a directory: {resolved_ood_root}")
            STATE["selected_ood_dataset_name"] = None
            return resolved_ood_root
        if requested_ood_dataset:
            selected_ood_dataset_name, selected_ood_root, available_ood_dataset_names = resolve_repo_dataset_directory(
                repo_root=ROOT,
                repo_relative_root=OOD_DATASET_ROOT,
                requested_name=requested_ood_dataset,
                prompt_label="OOD dataset",
            )
            print(f"[PREP] repo OOD dataset options={available_ood_dataset_names}")
            STATE["selected_ood_dataset_name"] = selected_ood_dataset_name
            return selected_ood_root
        if not ASK_FOR_OOD_ROOT:
            STATE["selected_ood_dataset_name"] = None
            return None
        requested_ood_root = str(input("OOD klasoru yolu girin (bos birak: OOD kopyalama): ")).strip()
        if requested_ood_root:
            resolved_ood_root = Path(requested_ood_root).expanduser()
            if not resolved_ood_root.is_absolute():
                resolved_ood_root = (ROOT / resolved_ood_root).resolve()
            if not resolved_ood_root.is_dir():
                raise RuntimeError(f"OOD root not found or not a directory: {resolved_ood_root}")
            STATE["selected_ood_dataset_name"] = None
            return resolved_ood_root
        STATE["selected_ood_dataset_name"] = None
        return None

    def _format_runtime_blockers(summary: dict, artifact_root: Path) -> str:
        counts = dict(summary.get("summary", {}) or {})
        blocking_issues = list(summary.get("blocking_issues") or [])
        lines = [
            "Audit sonucu bloklayici sorunlar iceriyor. Materyalizasyon once temizlenmeli.",
            f"artifact_root={artifact_root}",
            f"dataset_root={STATE.get('dataset_root')}",
            (
                f"runtime_ready={summary.get('runtime_ready')} "
                f"blocking_issues={counts.get('blocking_issues', len(blocking_issues))} "
                f"cross_class_conflicts={counts.get('cross_class_conflicts', 0)} "
                f"same_class_high_risk_clusters={counts.get('same_class_high_risk_clusters', 0)}"
            ),
        ]
        if blocking_issues:
            lines.append("blocking_issues:")
            for item in blocking_issues[:10]:
                lines.append(f" - {item}")
            if len(blocking_issues) > 10:
                lines.append(f" - ... {len(blocking_issues) - 10} more")
        lines.extend(
            [
                "inspect_reports:",
                f" - {artifact_root / 'prep_summary.json'}",
                f" - {artifact_root / 'class_health_report.json'}",
                f" - {artifact_root / 'cross_class_conflicts.csv'}",
                f" - {artifact_root / 'same_class_high_risk_clusters.csv'}",
            ]
        )
        return "\n".join(lines)

    release_mode = bool(str(DATASET_RELEASE_TAG or "").strip())
    if release_mode:
        release_report = fetch_materialize_dataset_release(
            root=ROOT,
            repository=DATASET_RELEASE_REPOSITORY,
            release_tag=DATASET_RELEASE_TAG,
            target=DATASET_RELEASE_TARGET,
            cache_root=DATASET_RELEASE_CACHE_ROOT,
        )
        STATE["dataset_release_report"] = release_report
        STATE["runtime_dataset_root"] = Path(release_report["runtime_dataset_root"])
        STATE["selected_dataset_root"] = Path(release_report["selected_dataset_root"])
        STATE["audit_summary"] = {"runtime_ready": True, "source": "github_release"}
        MATERIALIZE_AFTER_REVIEW = False
        print(
            "[DATASET RELEASE] "
            f"repository={release_report['repository']} tag={release_report['release_tag']} "
            f"target={release_report['target']} verified=True read_only=True"
        )
        print(f"[DATASET RELEASE] selected_dataset_root={release_report['selected_dataset_root']}")
        TELEMETRY.update_latest(
            {
                "phase": "dataset_release_materialized",
                "dataset_release": release_report,
            }
        )

    summary = STATE["audit_summary"]
    if STATE.get("human_review_stop_requested"):
        raise RuntimeError("Human review durdurma karari aktif. Audit artifactlerini inceleyip audit/prep hucrelerini yeniden calistirin.")
    review_packet = STATE.get("human_review_packet")
    if INTERACTIVE_AUDIT_REVIEW and review_packet and review_packet.get("pause_recommended") and not STATE.get("human_review_approved"):
        print(format_human_review_packet(review_packet))
        if not _prompt_yes_no("Bu audit karariyla runtime dataset materyalize edilsin mi?", True):
            STATE["human_review_stop_requested"] = True
            raise RuntimeError("Runtime materialization human review tarafindan durduruldu.")
        STATE["human_review_approved"] = True
    if not MATERIALIZE_AFTER_REVIEW:
        print("[PREP] MATERIALIZE_AFTER_REVIEW=False. Audit dosyalari incelemeye hazir.")
    else:
        if not summary.get("runtime_ready"):
            raise RuntimeError(_format_runtime_blockers(summary, Path(STATE["artifact_root"])))

        ood_root = _resolve_optional_ood_root()
        runtime_root = Path(PREPARED_RUNTIME_ROOT).expanduser()
        if not runtime_root.is_absolute():
            runtime_root = (ROOT / runtime_root).resolve()

        dataset_key = build_prepared_dataset_key(CROP_NAME, PART_NAME)
        runtime_dataset_root = materialize_grouped_runtime_dataset(
            class_root=STATE["dataset_root"],
            crop_name=CROP_NAME,
            part_name=PART_NAME,
            artifact_root=STATE["artifact_root"],
            runtime_root=runtime_root,
            ood_root=ood_root,
            materialization_strategy="copy",
        )

        STATE["runtime_dataset_root"] = runtime_dataset_root
        STATE["resolved_ood_root"] = str(ood_root) if ood_root is not None else ""
        print(f"[PREP] Hazir runtime dataset su klasore yazildi: {runtime_dataset_root / dataset_key}")

        TELEMETRY.update_latest({
            "phase": "data_prep_materialized",
            "runtime_dataset_root": str(runtime_dataset_root),
            "resolved_ood_root": str(STATE.get("resolved_ood_root") or ""),
            "dataset_source": str(STATE.get("dataset_source") or "repo"),
        })

    # Finalize notebook and telemetry
    notebook_export_result = export_current_colab_notebook(REPO_NOTEBOOK_OUTPUT_PATH)
    TELEMETRY.merge_summary_metadata({
        "access_check": STATE.get("access_report", {}),
        "prep_summary": STATE.get("audit_summary", {}),
        "materialization_prep_summary": STATE.get("prep_materialization_result", {}),
        "dataset_release_report": STATE.get("dataset_release_report", {}),
        "repo_run_dir": str(REPO_RUN_DIR),
        "notebook_export_path": str(notebook_export_result or REPO_NOTEBOOK_OUTPUT_PATH),
    })
    TELEMETRY.close({
        "status": "ok",
        "runtime_ready": bool((STATE.get("audit_summary") or {}).get("runtime_ready")),
        "materialized": bool(STATE.get("runtime_dataset_root")),
        "dataset_release_verified": bool((STATE.get("dataset_release_report") or {}).get("verified")),
        "repo_run_dir": str(REPO_RUN_DIR),
    })
    checkpoint_root = getattr(TELEMETRY, "drive_run_dir", None)
    if checkpoint_root is None or not Path(checkpoint_root).exists():
        checkpoint_root = getattr(TELEMETRY, "local_run_dir", REPO_RUN_DIR / "telemetry")
    assert checkpoint_root is not None
    REPO_RUN_EXPORTS.update(
        save_run_outputs_to_repo(
            LOCAL_OUTPUT_DIR=ROOT / "outputs" / "colab_notebook_data_prep",
            REPO_OUTPUT_DIR=REPO_RUN_DIR / "outputs" / "colab_notebook_data_prep",
            TELEMETRY=TELEMETRY,
            CHECKPOINT_ROOT=Path(checkpoint_root),
            REPO_TELEMETRY_DIR=REPO_RUN_DIR / "telemetry",
            REPO_CHECKPOINT_STATE_DIR=REPO_RUN_DIR / "checkpoint_state",
        )
    )
    for key in sorted(REPO_RUN_EXPORTS):
        print(f"[RUNS] {key} -> {REPO_RUN_EXPORTS[key]}")
    print(f"[RUNS] notebook -> {REPO_NOTEBOOK_OUTPUT_PATH}")


__all__ = [
    'run_bootstrap_notebook',
    'init_run_environment',
    'save_run_outputs_to_repo',
    'import_dataset_from_drive',
    '_drive_destination_parent',
    'resolve_dataset_release_manifest',
    'fetch_materialize_dataset_release',
    'run_dataset_audit',
    'run_prepare_dataset',
    'run_materialize_runtime_dataset',
]
