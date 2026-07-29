#!/usr/bin/env python3
"""Minimal Notebook 4 UI for direct adapter smoke testing."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from IPython.display import HTML, clear_output, display
from PIL import Image

from scripts.simple_adapter_smoke_ui_view import (
    _apply_dropdown_options as _apply_dropdown_options,
)
from scripts.simple_adapter_smoke_ui_view import (
    _build_adapter_details_html as _build_adapter_details_html,
)
from scripts.simple_adapter_smoke_ui_view import _build_help_text_html as _build_help_text_html
from scripts.simple_adapter_smoke_ui_view import _build_result_html as _build_result_html
from scripts.simple_adapter_smoke_ui_view import _display_error_box as _display_error_box
from scripts.simple_adapter_smoke_ui_view import _display_html as _display_html
from scripts.simple_adapter_smoke_ui_view import _error_box_html as _error_box_html
from scripts.simple_adapter_smoke_ui_view import _extract_upload_record as _extract_upload_record
from scripts.simple_adapter_smoke_ui_view import _hf_access_error_html as _hf_access_error_html
from scripts.simple_adapter_smoke_ui_view import _persist_upload_value as _persist_upload_value
from scripts.simple_adapter_smoke_ui_view import _raw_json_details_html as _raw_json_details_html
from scripts.simple_adapter_smoke_ui_view import _section_header_html as _section_header_html
from scripts.simple_adapter_smoke_ui_view import _show_status_message as _show_status_message

# Defer heavy imports until actually needed (when UI is launched and user interacts)
try:
    import ipywidgets as widgets
except Exception:  # pragma: no cover - notebook runtime fallback
    widgets = None

# Lazy import placeholders - will be populated on-demand
_build_prediction_visualization_images = None
_discover_adapter_candidates = None
_load_adapter_summary = None
_predict_single_image = None

# Public compatibility aliases for older tests and notebook copies.
build_prediction_visualization_images = None
discover_adapter_candidates = None
load_adapter_summary = None
predict_single_image = None

_PREDICTION_ERROR_TYPES = (
    FileNotFoundError,
    ValueError,
    RuntimeError,
    OSError,
    TypeError,
    KeyError,
    AttributeError,
)


def _resolve_notebook_device(requested_device: Any) -> tuple[str, Optional[str]]:
    requested = str(requested_device or "cpu").strip() or "cpu"
    if not requested.startswith("cuda"):
        return requested, None

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on notebook runtime
        return "cpu", f"Requested device {requested!r} but torch could not be imported; using 'cpu'. Error: {exc}"

    try:
        if not torch.cuda.is_available():
            return "cpu", f"Requested device {requested!r} but CUDA is not available; using 'cpu'."
        if ":" in requested:
            raw_index = requested.split(":", 1)[1]
            if raw_index:
                device_index = int(raw_index)
                device_count = int(torch.cuda.device_count())
                if device_index < 0 or device_index >= device_count:
                    fallback = "cuda:0" if device_count > 0 else "cpu"
                    return (
                        fallback,
                        f"Requested device {requested!r} but only {device_count} CUDA device(s) are visible; "
                        f"using {fallback!r}.",
                    )
    except Exception as exc:  # pragma: no cover - defensive notebook runtime guard
        return "cpu", f"Requested device {requested!r} could not be validated; using 'cpu'. Error: {exc}"

    return requested, None


def _ensure_adapter_smoke_imports():
    """Lazy import adapter smoke functions when needed."""
    global \
        _build_prediction_visualization_images, \
        _discover_adapter_candidates, \
        _load_adapter_summary, \
        _predict_single_image
    global \
        build_prediction_visualization_images, \
        discover_adapter_candidates, \
        load_adapter_summary, \
        predict_single_image
    if _build_prediction_visualization_images is None:
        from src.pipeline.adapter_smoke import (
            build_prediction_visualization_images as _bpvi,
        )
        from src.pipeline.adapter_smoke import (
            discover_adapter_candidates as _dac,
        )
        from src.pipeline.adapter_smoke import (
            load_adapter_summary as _las,
        )
        from src.pipeline.adapter_smoke import (
            predict_single_image as _psi,
        )

        _build_prediction_visualization_images = build_prediction_visualization_images or _bpvi
        _discover_adapter_candidates = discover_adapter_candidates or _dac
        _load_adapter_summary = load_adapter_summary or _las
        _predict_single_image = predict_single_image or _psi
        if build_prediction_visualization_images is None:
            build_prediction_visualization_images = _build_prediction_visualization_images
        if discover_adapter_candidates is None:
            discover_adapter_candidates = _discover_adapter_candidates
        if load_adapter_summary is None:
            load_adapter_summary = _load_adapter_summary
        if predict_single_image is None:
            predict_single_image = _predict_single_image


@lru_cache(maxsize=16)
def _cached_discover_adapter_candidates(
    search_roots_key: tuple[str, ...],
    crop_name: Optional[str],
    collapse_run_mirrors: bool,
    discovery_source_token: int,
    refresh_nonce: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        discover_adapter_candidates(
            [Path(candidate) for candidate in search_roots_key],
            crop_name=crop_name,
            collapse_run_mirrors=collapse_run_mirrors,
        )
    )


def _discover_adapter_candidates_for_ui(
    search_roots_key: tuple[str, ...],
    *,
    collapse_run_mirrors: bool,
    discovery_token: int,
    refresh_nonce: int,
) -> list[dict[str, Any]]:
    return list(
        _cached_discover_adapter_candidates(
            search_roots_key,
            None,
            collapse_run_mirrors,
            discovery_token,
            refresh_nonce,
        )
    )


def _adapter_dropdown_options(adapter_candidates: list[dict[str, Any]]) -> list[tuple[str, int]]:
    options = [(candidate["display_name"], index) for index, candidate in enumerate(adapter_candidates)]
    return options or [("Adapter bulunamadi, asagidan yol girin", -1)]


def _default_adapter_search_roots(root_path: Path, *, include_run_adapters: bool = False) -> list[Path]:
    roots = [
        root_path / "outputs" / "colab_notebook_training",
        root_path / "outputs" / "colab_notebook_training" / "telemetry_runtime" / "telemetry",
        root_path / "models" / "adapters",
    ]
    if include_run_adapters:
        roots.append(root_path / "runs")
    return roots


def _load_adapter_summary_for_candidate(
    candidate: dict[str, Any],
    *,
    config_env: str,
    device: str,
) -> dict[str, Any]:
    return load_adapter_summary(
        candidate.get("crop_name"),
        adapter_dir=candidate.get("adapter_dir"),
        config_env=config_env,
        device=device,
    )


def _running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_colab_widget_manager() -> bool:
    """Backward-compatible helper kept for existing tests and callers."""
    try:
        from google.colab import output as colab_output
    except Exception:
        return False

    enable_manager = getattr(colab_output, "enable_custom_widget_manager", None)
    if enable_manager is None:
        return False

    enable_manager()
    return True


def _is_huggingface_gated_access_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    return (
        "gated repo" in lowered
        or "401 unauthorized" in lowered
        or "access to model" in lowered
        and "restricted" in lowered
        or "you must have access to it and be authenticated" in lowered
    )


def _upload_via_colab_files(upload_dir: Path) -> Optional[Path]:
    try:
        from google.colab import files
    except Exception:
        return None

    uploaded = files.upload()
    if not uploaded:
        return None
    upload_name, upload_bytes = next(iter(uploaded.items()))
    target_path = upload_dir / Path(str(upload_name)).name
    target_path.write_bytes(bytes(upload_bytes))
    return target_path


def launch_simple_adapter_smoke_ui(
    root: str | Path,
    *,
    search_roots: Optional[list[str | Path]] = None,
    show_all_adapters: bool = False,
    show_mirror_adapters: bool = False,
    config_env: str = "colab",
    device: str = "cuda",
    upload_dir_name: str = "notebook4_uploads",
    enable_prediction_visualization: bool = True,
    explanation_method: str = "attention_map",
    explanation_grid_size: int = 7,
    include_run_adapters: bool = False,
) -> None:
    """Render the minimal direct-adapter smoke-test UI used by Notebook 4.

    ``show_all_adapters`` is kept for older notebook copies and includes
    historical ``runs/`` exports. Mirror exports are hidden unless
    ``show_mirror_adapters`` is explicitly enabled.
    """
    if widgets is None:
        raise RuntimeError(
            "This notebook UI requires ipywidgets. Re-run the bootstrap cell after dependency installation."
        )

    root_path = Path(root)
    resolved_device, device_warning = _resolve_notebook_device(device)
    device = resolved_device
    resolved_search_roots = [
        Path(candidate)
        for candidate in (
            search_roots
            or _default_adapter_search_roots(
                root_path,
                include_run_adapters=include_run_adapters or show_all_adapters,
            )
        )
    ]
    upload_dir = root_path / ".runtime_tmp" / upload_dir_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    _ensure_adapter_smoke_imports()
    adapter_candidates_key = tuple(str(candidate) for candidate in resolved_search_roots)
    collapse_run_mirrors = not show_mirror_adapters
    discovery_token = id(discover_adapter_candidates)
    discovery_nonce = 0
    adapter_candidates = _discover_adapter_candidates_for_ui(
        adapter_candidates_key,
        collapse_run_mirrors=collapse_run_mirrors,
        discovery_token=discovery_token,
        refresh_nonce=discovery_nonce,
    )
    dropdown_options = _adapter_dropdown_options(adapter_candidates)

    title = widgets.HTML('<h3 style="margin:0 0 8px 0;">Basit Adapter Testi</h3>')
    help_text = widgets.HTML(_build_help_text_html(_running_in_colab()))
    adapter_dropdown = widgets.Dropdown(
        options=[],
        description="Adapter:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    _apply_dropdown_options(adapter_dropdown, dropdown_options)
    adapter_path_text = widgets.Text(
        value="",
        placeholder="Isterseniz ADAPTER_DIR veya adapter_meta.json yolu girin",
        description="Yol:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    image_path_text = widgets.Text(
        value="",
        placeholder="Mevcut dosya yolunu girin veya asagidaki yukleme dugmesini kullanin",
        description="Resim:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    upload_widget = widgets.FileUpload(
        accept="image/*",
        multiple=False,
        description="Resim Yukle",
        layout=widgets.Layout(width="180px"),
    )
    visualization_checkbox = widgets.Checkbox(
        value=bool(enable_prediction_visualization),
        description="Gorsel aciklama",
        indent=False,
        layout=widgets.Layout(width="95%"),
    )
    explanation_method_dropdown = widgets.Dropdown(
        options=[
            ("Attention map", "attention_map"),
            ("Occlusion sensitivity", "occlusion_sensitivity"),
        ],
        value=str(explanation_method),
        description="Yontem:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    refresh_button = widgets.Button(description="Adapterleri Yenile", button_style="info")
    clear_image_button = widgets.Button(description="Yeni Resim", button_style="warning")
    run_button = widgets.Button(description="Tahmin Et", button_style="success")
    status_output = widgets.Output()
    adapter_details_output = widgets.Output()
    result_output = widgets.Output()

    def selected_candidate() -> dict[str, Any]:
        manual_path = adapter_path_text.value.strip()
        if manual_path:
            return {"adapter_dir": manual_path, "crop_name": None, "display_name": manual_path}

        selected_index = int(adapter_dropdown.value)
        if selected_index < 0 or selected_index >= len(adapter_candidates):
            raise FileNotFoundError("Adapter bulunamadi. Ya yol girin ya da search_roots altinda adapter bulundurun.")
        return adapter_candidates[selected_index]

    def resolve_image_path() -> Path:
        raw_path = image_path_text.value.strip()
        if raw_path:
            return Path(raw_path).expanduser()
        if _running_in_colab():
            uploaded_path = _upload_via_colab_files(upload_dir)
            if uploaded_path is not None:
                image_path_text.value = str(uploaded_path)
                _show_status_message(status_output, f"Yuklenen resim hazir: {uploaded_path.name}")
                return uploaded_path
            raise ValueError("Colab upload iptal edildi veya dosya secilmedi.")
        raise ValueError("Bir resim yolu girin. Colab'da bos birakirsaniz upload penceresi acilir.")

    def handle_widget_upload(change: Any = None) -> None:
        uploaded_path = _persist_upload_value(upload_widget.value, upload_dir)
        if uploaded_path is None:
            return
        image_path_text.value = str(uploaded_path)
        _show_status_message(status_output, f"Yuklenen resim hazir: {uploaded_path.name}")

    def clear_image(_button: Any = None) -> None:
        image_path_text.value = ""
        _show_status_message(status_output, "Yeni resim yukleyin veya Resim alanina dosya yolu girin.")

    def render_adapter_details() -> None:
        try:
            _ensure_adapter_smoke_imports()
            candidate = selected_candidate()
            summary = _load_adapter_summary_for_candidate(candidate, config_env=config_env, device=device)
            _display_html(adapter_details_output, _build_adapter_details_html(summary))
        except Exception as exc:
            _display_error_box(adapter_details_output, "Adapter bilgisi yuklenemedi:", str(exc))

    def refresh(_button: Any = None) -> None:
        nonlocal adapter_candidates, discovery_nonce
        _show_status_message(status_output, "Adapter listesi yenileniyor...")
        _ensure_adapter_smoke_imports()
        discovery_nonce += 1
        adapter_candidates = _discover_adapter_candidates_for_ui(
            adapter_candidates_key,
            collapse_run_mirrors=collapse_run_mirrors,
            discovery_token=id(discover_adapter_candidates),
            refresh_nonce=discovery_nonce,
        )
        options = _adapter_dropdown_options(adapter_candidates)
        adapter_dropdown.options = options
        adapter_dropdown.value = options[0][1]
        render_adapter_details()
        _show_status_message(
            status_output,
            f"Bulunan adapter sayisi: {len(adapter_candidates)}",
            f"Cihaz: {device}",
            *([device_warning] if device_warning else []),
            *[f"- taranan kok: {candidate_root}" for candidate_root in resolved_search_roots],
        )

    def run_prediction(_button: Any = None) -> None:
        with result_output:
            clear_output(wait=True)
            try:
                _ensure_adapter_smoke_imports()
                predict_single_image_fn = predict_single_image
                build_prediction_visualization_images_fn = build_prediction_visualization_images
                image_path = resolve_image_path()
                if not image_path.exists():
                    raise FileNotFoundError(f"Resim bulunamadi: {image_path}")
                candidate = selected_candidate()
                summary = _load_adapter_summary_for_candidate(candidate, config_env=config_env, device=device)
                required_keys = {"crop_name", "resolved_adapter_dir"}
                missing_keys = required_keys - set(summary.keys())
                if missing_keys:
                    raise ValueError(
                        f"Adapter summary is missing required keys: {missing_keys}. Got keys: {set(summary.keys())}."
                    )
                with Image.open(image_path) as preview:
                    display(preview.copy())
                result = predict_single_image_fn(
                    image_path,
                    summary["crop_name"],
                    adapter_dir=summary["resolved_adapter_dir"],
                    config_env=config_env,
                    device=device,
                    enable_robust_smoke=True,
                    explain_prediction=bool(visualization_checkbox.value),
                    explanation_grid_size=int(explanation_grid_size),
                    explanation_method=str(explanation_method_dropdown.value),
                )
                visualization_images = build_prediction_visualization_images_fn(image_path, result)
                if visualization_images:
                    display(HTML(_section_header_html("Model gorunumu ve aciklama haritasi")))
                    display(visualization_images["model_view"])
                    display(visualization_images["heatmap_overlay"])
                _display_html(result_output, _build_result_html(summary, result, image_path))
                display(HTML(_raw_json_details_html(result)))
            except Exception as exc:
                if _is_huggingface_gated_access_error(exc):
                    display(HTML(_hf_access_error_html(exc)))
                    return
                if not isinstance(exc, _PREDICTION_ERROR_TYPES):
                    raise
                _display_error_box(result_output, "Tahmin hatasi:", str(exc))

    refresh_button.on_click(refresh)
    clear_image_button.on_click(clear_image)
    run_button.on_click(run_prediction)
    adapter_dropdown.observe(lambda change: render_adapter_details(), names="value")
    adapter_path_text.observe(lambda change: render_adapter_details(), names="value")
    upload_widget.observe(handle_widget_upload, names="value")
    display(
        widgets.VBox(
            [
                title,
                help_text,
                adapter_dropdown,
                adapter_path_text,
                adapter_details_output,
                image_path_text,
                widgets.HBox([upload_widget, clear_image_button]),
                visualization_checkbox,
                explanation_method_dropdown,
                widgets.HBox([refresh_button, run_button]),
                status_output,
                result_output,
            ]
        )
    )
    refresh()


__all__ = [
    "launch_simple_adapter_smoke_ui",
    "_running_in_colab",
    "_ensure_colab_widget_manager",
    "_resolve_notebook_device",
    "_build_result_html",
    "_persist_upload_value",
    "_upload_via_colab_files",
    "_is_huggingface_gated_access_error",
    "_hf_access_error_html",
]
