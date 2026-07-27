import sys
import types
from pathlib import Path

from scripts import colab_simple_adapter_smoke_ui as ui


def test_ensure_colab_widget_manager_returns_false_outside_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert ui._ensure_colab_widget_manager() is False


def test_ensure_colab_widget_manager_enables_custom_manager(monkeypatch):
    calls: list[str] = []

    output_module = types.ModuleType("google.colab.output")
    output_module.enable_custom_widget_manager = lambda: calls.append("enabled")

    colab_module = types.ModuleType("google.colab")
    colab_module.output = output_module

    google_module = types.ModuleType("google")
    google_module.colab = colab_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setitem(sys.modules, "google.colab.output", output_module)

    assert ui._ensure_colab_widget_manager() is True
    assert calls == ["enabled"]


def test_resolve_notebook_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device, warning = ui._resolve_notebook_device("cuda")

    assert device == "cpu"
    assert "CUDA is not available" in str(warning)


def test_resolve_notebook_device_falls_back_from_missing_cuda_index(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device, warning = ui._resolve_notebook_device("cuda:7")

    assert device == "cuda:0"
    assert "only 1 CUDA device" in str(warning)


def test_default_adapter_search_roots_excludes_historical_runs(tmp_path: Path):
    roots = ui._default_adapter_search_roots(tmp_path)

    assert roots == [
        tmp_path / "outputs" / "colab_notebook_training",
        tmp_path / "outputs" / "colab_notebook_training" / "telemetry_runtime" / "telemetry",
        tmp_path / "models" / "adapters",
    ]


def test_default_adapter_search_roots_can_include_runs_for_debug(tmp_path: Path):
    roots = ui._default_adapter_search_roots(tmp_path, include_run_adapters=True)

    assert roots[-1] == tmp_path / "runs"


def test_launch_simple_adapter_smoke_ui_builds_minimal_layout(monkeypatch, tmp_path):
    class _FakeWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on_click(self, _handler):
            return None

        def observe(self, _handler, names=None):
            return None

    fake_widgets = types.SimpleNamespace(
        HTML=_FakeWidget,
        Dropdown=_FakeWidget,
        Text=_FakeWidget,
        FileUpload=_FakeWidget,
        Checkbox=_FakeWidget,
        Button=_FakeWidget,
        Output=_FakeWidget,
        VBox=_FakeWidget,
        HBox=_FakeWidget,
        Layout=lambda **kwargs: kwargs,
    )

    monkeypatch.setattr(ui, "widgets", fake_widgets)
    monkeypatch.setattr(ui, "_running_in_colab", lambda: True)
    monkeypatch.setattr(ui, "discover_adapter_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ui, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "clear_output", lambda *_args, **_kwargs: None)

    ui.launch_simple_adapter_smoke_ui(tmp_path)


def test_launch_simple_adapter_smoke_ui_uses_resolved_cpu_device(monkeypatch, tmp_path):
    class _FakeWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on_click(self, _handler):
            return None

        def observe(self, _handler, names=None):
            return None

    fake_widgets = types.SimpleNamespace(
        HTML=_FakeWidget,
        Dropdown=_FakeWidget,
        Text=_FakeWidget,
        FileUpload=_FakeWidget,
        Checkbox=_FakeWidget,
        Button=_FakeWidget,
        Output=_FakeWidget,
        VBox=_FakeWidget,
        HBox=_FakeWidget,
        Layout=lambda **kwargs: kwargs,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    seen_devices: list[str] = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(ui, "widgets", fake_widgets)
    monkeypatch.setattr(ui, "_running_in_colab", lambda: True)
    monkeypatch.setattr(ui, "_ensure_adapter_smoke_imports", lambda: None)
    monkeypatch.setattr(
        ui,
        "discover_adapter_candidates",
        lambda *_args, **_kwargs: [
            {
                "display_name": "tomato | leaf",
                "crop_name": "tomato",
                "adapter_dir": str(tmp_path / "adapter"),
            }
        ],
    )

    def _fake_load_summary(_crop_name, *, adapter_dir=None, config_env, device):
        seen_devices.append(device)
        return {
            "crop_name": "tomato",
            "part_name": "leaf",
            "resolved_adapter_dir": str(tmp_path / "adapter"),
            "class_names": ["healthy"],
        }

    monkeypatch.setattr(ui, "load_adapter_summary", _fake_load_summary)
    monkeypatch.setattr(ui, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "clear_output", lambda *_args, **_kwargs: None)
    ui._cached_discover_adapter_candidates.cache_clear()

    ui.launch_simple_adapter_smoke_ui(tmp_path, device="cuda")

    assert seen_devices == ["cpu"]


def test_launch_simple_adapter_smoke_ui_collapses_run_mirrors_by_default(monkeypatch, tmp_path):
    class _FakeWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on_click(self, _handler):
            return None

        def observe(self, _handler, names=None):
            return None

    fake_widgets = types.SimpleNamespace(
        HTML=_FakeWidget,
        Dropdown=_FakeWidget,
        Text=_FakeWidget,
        FileUpload=_FakeWidget,
        Checkbox=_FakeWidget,
        Button=_FakeWidget,
        Output=_FakeWidget,
        VBox=_FakeWidget,
        HBox=_FakeWidget,
        Layout=lambda **kwargs: kwargs,
    )
    calls: list[dict[str, object]] = []

    def _fake_discover(*_args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(ui, "widgets", fake_widgets)
    monkeypatch.setattr(ui, "_running_in_colab", lambda: True)
    monkeypatch.setattr(ui, "discover_adapter_candidates", _fake_discover)
    monkeypatch.setattr(ui, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "clear_output", lambda *_args, **_kwargs: None)

    ui.launch_simple_adapter_smoke_ui(tmp_path)

    assert calls[0]["collapse_run_mirrors"] is True


def test_launch_simple_adapter_smoke_ui_keeps_legacy_show_all_collapsed(monkeypatch, tmp_path):
    class _FakeWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on_click(self, _handler):
            return None

        def observe(self, _handler, names=None):
            return None

    fake_widgets = types.SimpleNamespace(
        HTML=_FakeWidget,
        Dropdown=_FakeWidget,
        Text=_FakeWidget,
        FileUpload=_FakeWidget,
        Checkbox=_FakeWidget,
        Button=_FakeWidget,
        Output=_FakeWidget,
        VBox=_FakeWidget,
        HBox=_FakeWidget,
        Layout=lambda **kwargs: kwargs,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_discover(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(ui, "widgets", fake_widgets)
    monkeypatch.setattr(ui, "_running_in_colab", lambda: True)
    monkeypatch.setattr(ui, "discover_adapter_candidates", _fake_discover)
    monkeypatch.setattr(ui, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "clear_output", lambda *_args, **_kwargs: None)

    ui.launch_simple_adapter_smoke_ui(tmp_path, show_all_adapters=True)

    assert calls[0][1]["collapse_run_mirrors"] is True
    assert tmp_path / "runs" in calls[0][0][0]


def test_launch_simple_adapter_smoke_ui_can_show_mirrors_for_debug(monkeypatch, tmp_path):
    class _FakeWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on_click(self, _handler):
            return None

        def observe(self, _handler, names=None):
            return None

    fake_widgets = types.SimpleNamespace(
        HTML=_FakeWidget,
        Dropdown=_FakeWidget,
        Text=_FakeWidget,
        FileUpload=_FakeWidget,
        Checkbox=_FakeWidget,
        Button=_FakeWidget,
        Output=_FakeWidget,
        VBox=_FakeWidget,
        HBox=_FakeWidget,
        Layout=lambda **kwargs: kwargs,
    )
    calls: list[dict[str, object]] = []

    def _fake_discover(*_args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(ui, "widgets", fake_widgets)
    monkeypatch.setattr(ui, "_running_in_colab", lambda: True)
    monkeypatch.setattr(ui, "discover_adapter_candidates", _fake_discover)
    monkeypatch.setattr(ui, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "clear_output", lambda *_args, **_kwargs: None)

    ui.launch_simple_adapter_smoke_ui(tmp_path, show_mirror_adapters=True)

    assert calls[0]["collapse_run_mirrors"] is False


def test_build_result_html_sets_explicit_contrast_colors():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "predicted_class": "septoria",
            "confidence": 0.9712,
            "is_ood": False,
            "primary_score": 0.8673,
            "decision_threshold": 1.8515,
            "view_consistency": {"stable": True, "warning_codes": []},
            "uncertainty_diagnostics": {"warning_codes": []},
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "background:#ffffff;color:#111827" in html
    assert "summary style=\"cursor:pointer;color:#111827;font-weight:600;\"" in html
    assert "Stable across derived views" in html
    assert "Status:</b> unknown" in html


def test_build_result_html_mentions_occlusion_visualization_when_available():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "status": "success",
            "predicted_class": "septoria",
            "confidence": 0.9712,
            "is_ood": False,
            "primary_score": 0.8673,
            "decision_threshold": 1.8515,
            "view_consistency": {"stable": True, "warning_codes": []},
            "uncertainty_diagnostics": {"warning_codes": []},
            "visualization": {
                "method": "occlusion_sensitivity",
                "view_name": "full_resize",
                "heatmap": [[0.0, 1.0], [0.25, 0.5]],
            },
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "Gorsel Aciklama" in html
    assert "occlusion sensitivity haritasi hazir" in html


def test_build_result_html_mentions_attention_visualization_when_available():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "status": "success",
            "predicted_class": "septoria",
            "confidence": 0.9712,
            "is_ood": False,
            "primary_score": 0.8673,
            "decision_threshold": 1.8515,
            "view_consistency": {"stable": True, "warning_codes": []},
            "uncertainty_diagnostics": {"warning_codes": []},
            "visualization": {
                "status": "success",
                "method": "attention_map",
                "view_name": "full_resize",
                "heatmap": [[0.0, 1.0], [0.25, 0.5]],
            },
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "Gorsel Aciklama" in html
    assert "attention map hazir" in html


def test_build_result_html_mentions_unavailable_visualization():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "status": "success",
            "predicted_class": "septoria",
            "confidence": 0.9712,
            "is_ood": False,
            "primary_score": 0.8673,
            "decision_threshold": 1.8515,
            "view_consistency": {"stable": True, "warning_codes": []},
            "uncertainty_diagnostics": {"warning_codes": []},
            "visualization": {
                "status": "unavailable",
                "method": "attention_map",
                "error": "The loaded backbone did not return attentions.",
            },
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "hazirlanamadi" in html
    assert "attention_map" in html


def test_build_result_html_surfaces_error_state_without_in_distribution_label():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "status": "error",
            "predicted_class": None,
            "confidence": 0.0,
            "is_ood": None,
            "primary_score": None,
            "decision_threshold": None,
            "error": "forced failure on primary view",
            "view_consistency": {"stable": False, "warning_codes": ["view_error_present"]},
            "uncertainty_diagnostics": {"warning_codes": ["prediction_error"]},
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "Tahmin Basarisiz" in html
    assert "Status:</b> error" in html
    assert "OOD Karari:</b> -" in html
    assert "Prediction failed" in html
    assert "Hata:</b> forced failure on primary view" in html
    assert "In-distribution" not in html


def test_build_result_html_explains_robustness_warning_codes():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "grape",
        },
        {
            "status": "success",
            "predicted_class": "botrytis_bunch_rot",
            "confidence": 0.6829,
            "is_ood": False,
            "primary_score": 0.4000,
            "decision_threshold": 1.4792,
            "view_consistency": {
                "stable": False,
                "warning_codes": ["view_class_disagreement", "view_confidence_spread_high"],
                "predicted_classes": {
                    "full_resize": "botrytis_bunch_rot",
                    "resize_pad": "healthy",
                },
                "ood_votes": {
                    "full_resize": False,
                    "resize_pad": False,
                },
                "confidence_min": 0.44,
                "confidence_max": 0.6829,
                "confidence_spread": 0.2429,
            },
            "uncertainty_diagnostics": {
                "warning_codes": ["confidence_not_calibrated", "view_instability"],
            },
        },
        image_path=ui.Path("/tmp/grape.png"),
    )

    assert "Robustluk Aciklamasi" in html
    assert "farkli gorunumler farkli siniflar tahmin etti" in html
    assert "full_resize: botrytis_bunch_rot; resize_pad: healthy" in html
    assert "gorunumler arasindaki confidence farki yuksek" in html
    assert "fark 24.29 puan" in html
    assert "kalibre edilmis olasilik gibi yorumlanmamalidir" in html
    assert "robust gorunumler stabil degil" in html


def test_detects_huggingface_gated_access_errors():
    error = OSError(
        "You are trying to access a gated repo. 401 Client Error. "
        "Access to model facebook/dinov3-vitl16-pretrain-lvd1689m is restricted."
    )

    assert ui._is_huggingface_gated_access_error(error) is True


def test_hf_access_error_html_guides_colab_secret_setup():
    html = ui._hf_access_error_html(OSError("Cannot access gated repo"))

    assert "HF_TOKEN" in html
    assert "Runtime'i yeniden baslatin" in html
    assert "Cannot access gated repo" in html


def test_persist_upload_value_writes_uploaded_bytes(tmp_path: Path):
    persisted = ui._persist_upload_value(
        (
            {
                "name": "leaf.png",
                "content": memoryview(b"abc123"),
            },
        ),
        tmp_path,
    )

    assert persisted == tmp_path / "leaf.png"
    assert persisted.read_bytes() == b"abc123"


def test_persist_upload_value_returns_none_for_empty_upload(tmp_path: Path):
    assert ui._persist_upload_value((), tmp_path) is None


def test_upload_via_colab_files_writes_uploaded_bytes(monkeypatch, tmp_path: Path):
    files_module = types.SimpleNamespace(upload=lambda: {"leaf.png": b"xyz"})
    colab_module = types.ModuleType("google.colab")
    colab_module.files = files_module
    google_module = types.ModuleType("google")
    google_module.colab = colab_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)

    persisted = ui._upload_via_colab_files(tmp_path)

    assert persisted == tmp_path / "leaf.png"
    assert persisted.read_bytes() == b"xyz"


def test_upload_via_colab_files_returns_none_outside_colab(monkeypatch, tmp_path: Path):
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert ui._upload_via_colab_files(tmp_path) is None
