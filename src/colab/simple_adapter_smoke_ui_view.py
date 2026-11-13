#!/usr/bin/env python3
"""HTML rendering and upload helpers for the Notebook 4 adapter smoke UI."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Optional

from IPython.display import HTML, clear_output, display


def _show_status_message(output_widget: Any, *messages: str) -> None:
    with output_widget:
        clear_output(wait=True)
        for message in messages:
            print(message)


def _error_box_html(
    title: str,
    message: str,
    *,
    border_color: str = "#fecaca",
    background_color: str = "#fff7f7",
    text_color: str = "#991b1b",
) -> str:
    return (
        f'<div style="border:1px solid {border_color};border-radius:10px;padding:12px;margin-top:12px;'
        f'background:{background_color};color:{text_color};">'
        f'<b>{escape(title)}</b><pre style="white-space:pre-wrap;word-break:break-word;">'
        f"{escape(message)}</pre></div>"
    )


def _display_html(output_widget: Any, html_text: str) -> None:
    with output_widget:
        clear_output(wait=True)
        display(HTML(html_text))


def _section_header_html(title: str) -> str:
    return f'<div style="margin-top:12px;font-weight:700;color:#111827;">{escape(title)}</div>'


def _raw_json_details_html(payload: dict[str, Any]) -> str:
    return "<details><summary>Ham JSON</summary><pre>" + json.dumps(payload, indent=2) + "</pre></details>"


def _display_error_box(output_widget: Any, title: str, message: str) -> None:
    _display_html(output_widget, _error_box_html(title, message))


def _apply_dropdown_options(dropdown: Any, options: list[tuple[str, int]]) -> None:
    dropdown.options = options
    dropdown.value = options[0][1]


def _build_help_text_html(running_in_colab: bool) -> str:
    help_parts = ["Adapter secin veya yol girin."]
    if running_in_colab:
        help_parts.append(
            "Yeni tahmin icin hucreyi tekrar calistirmadan resim yukleyin veya mevcut resim yolunu degistirin."
        )
    else:
        help_parts.append("Colab disinda <b>Resim</b> alanina mevcut dosya yolunu girin.")
    return '<p style="margin:0 0 12px 0;">' + " ".join(help_parts) + "</p>"


def _extract_upload_record(upload_value: Any) -> tuple[Optional[str], Optional[bytes]]:
    if isinstance(upload_value, dict):
        records = list(upload_value.values())
    elif isinstance(upload_value, (list, tuple)):
        records = list(upload_value)
    else:
        records = []
    if not records:
        return None, None

    record = records[0]
    if isinstance(record, dict):
        name = record.get("name", "uploaded_image")
        content = record.get("content", b"")
    else:
        name = getattr(record, "name", "uploaded_image")
        content = getattr(record, "content", b"")

    if isinstance(content, memoryview):
        content = content.tobytes()
    elif hasattr(content, "tobytes"):
        content = content.tobytes()
    return str(name), bytes(content)


def _persist_upload_value(upload_value: Any, upload_dir: Path) -> Optional[Path]:
    upload_name, upload_bytes = _extract_upload_record(upload_value)
    if not upload_name or not upload_bytes:
        return None
    target_path = upload_dir / Path(upload_name).name
    target_path.write_bytes(upload_bytes)
    return target_path


def _format_optional_float(value: Any, *, scale: float = 1.0, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * scale:.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _format_mapping_items(mapping: Any, *, value_formatter: Optional[Any] = None) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return "-"
    parts: list[str] = []
    for key, value in mapping.items():
        rendered_value = value_formatter(value) if value_formatter else str(value)
        parts.append(f"{key}: {rendered_value}")
    return "; ".join(parts)


def _bool_vote_label(value: Any) -> str:
    if value is True:
        return "OOD"
    if value is False:
        return "In-distribution"
    return "-"


def _warning_detail_items(
    view_consistency: dict[str, Any],
    uncertainty: dict[str, Any],
) -> list[str]:
    view_warning_codes = [str(code) for code in list(view_consistency.get("warning_codes", []))]
    uncertainty_warning_codes = [str(code) for code in list(uncertainty.get("warning_codes", []))]
    items: list[str] = [
        "Robust smoke, ayni gorseli farkli on-isleme gorunumleriyle tekrar tahmin eder; sonuc degisiyorsa tek tahmine guvenmeden goruntuyu ve adapteri manuel incelemek gerekir."
    ]

    if bool(view_consistency.get("stable")):
        items.append(
            "Stabil: uretilen gorunumler ayni sinifi ve ayni OOD kararini verdi; belirgin confidence farki veya gorunum hatasi yok."
        )
    elif not view_warning_codes and not uncertainty_warning_codes:
        items.append(
            "Inceleme onerisi var ama ayrintili uyari kodu gelmedi; ham JSON altindan views alanini kontrol edin."
        )

    view_warning_explanations = {
        "view_class_disagreement": (
            "view_class_disagreement: farkli gorunumler farkli siniflar tahmin etti. "
            "Gorunum bazli siniflar: " + _format_mapping_items(view_consistency.get("predicted_classes"))
        ),
        "view_ood_disagreement": (
            "view_ood_disagreement: gorunumler OOD kararinda ayrildi. "
            "Gorunum bazli kararlar: "
            + _format_mapping_items(view_consistency.get("ood_votes"), value_formatter=_bool_vote_label)
        ),
        "view_confidence_spread_high": (
            "view_confidence_spread_high: gorunumler arasindaki confidence farki yuksek. "
            "Min "
            + _format_optional_float(view_consistency.get("confidence_min"), scale=100.0, suffix="%")
            + ", max "
            + _format_optional_float(view_consistency.get("confidence_max"), scale=100.0, suffix="%")
            + ", fark "
            + _format_optional_float(view_consistency.get("confidence_spread"), scale=100.0, suffix=" puan")
            + "."
        ),
        "view_error_present": (
            "view_error_present: en az bir gorunum tahmini hata verdi. Hata veren gorunumler: "
            + (", ".join(str(name) for name in view_consistency.get("failed_views", [])) or "-")
        ),
    }
    uncertainty_warning_explanations = {
        "prediction_error": "prediction_error: ana gorunum tahmini hata verdi; hata satirini ve ham JSON'u kontrol edin.",
        "confidence_not_calibrated": (
            "confidence_not_calibrated: confidence top-1 softmax degeridir; kalibre edilmis olasilik gibi yorumlanmamalidir."
        ),
        "ood_flagged": "ood_flagged: ana gorunum adapter esigine gore OOD olarak isaretlendi.",
        "sure_confidence_reject": "sure_confidence_reject: daha siki confidence kontrolu tahmini reddetti.",
        "conformal_set_wide": (
            "conformal_set_wide: birden fazla makul sinif var; conformal set: "
            + (", ".join(str(item) for item in uncertainty.get("conformal_set", [])) or "-")
        ),
        "view_instability": "view_instability: robust gorunumler stabil degil; yukaridaki view uyarilari karar nedenini gosterir.",
    }

    for code in view_warning_codes:
        items.append(
            view_warning_explanations.get(code, f"{code}: tanimli olmayan view uyarisi; ham JSON'u kontrol edin.")
        )
    for code in uncertainty_warning_codes:
        items.append(
            uncertainty_warning_explanations.get(
                code,
                f"{code}: tanimli olmayan belirsizlik uyarisi; ham JSON'u kontrol edin.",
            )
        )

    return items


def _warning_details_html(view_consistency: dict[str, Any], uncertainty: dict[str, Any]) -> str:
    rows = "\n".join(
        f'<li style="margin:4px 0;">{escape(item)}</li>'
        for item in _warning_detail_items(view_consistency, uncertainty)
    )
    return f'<ul style="margin:6px 0 0 18px;padding:0;color:#374151;">{rows}</ul>'


def _hf_access_error_html(exc: BaseException) -> str:
    return f"""
    <div style="border:1px solid #fecaca;border-radius:10px;padding:14px;margin-top:12px;background:#fff7f7;color:#111827;">
      <div style="font-size:17px;font-weight:700;color:#991b1b;margin-bottom:8px;">Hugging Face model erisimi gerekli</div>
      <div>Secilen adapter gated backbone kullanıyor. Tahmin icin Colab secret olarak <b>HF_TOKEN</b> ekleyin.</div>
      <ol style="margin:8px 0 0 20px;padding:0;">
        <li>Hugging Face hesabinizda model erisimini onaylayin.</li>
        <li>Colab sol panel Secrets bolumune <b>HF_TOKEN</b> ekleyin.</li>
        <li>Runtime'i yeniden baslatin ve notebook bootstrap hucresini tekrar calistirin.</li>
      </ol>
      <details style="margin-top:10px;">
        <summary style="cursor:pointer;color:#111827;font-weight:600;">Teknik hata</summary>
        <pre style="white-space:pre-wrap;word-break:break-word;color:#7f1d1d;">{escape(str(exc))}</pre>
      </details>
    </div>
    """


def _build_result_html(summary: dict[str, Any], result: dict[str, Any], image_path: Path) -> str:
    status = str(result.get("status", "")).strip().lower() or "unknown"
    confidence_value = result.get("confidence")
    confidence_text = "-"
    if confidence_value is not None and status != "error":
        confidence_text = f"{float(confidence_value) * 100.0:.2f}%"
    is_ood = result.get("is_ood")
    if status == "error":
        ood_label = "-"
    elif is_ood is True:
        ood_label = "OOD"
    elif is_ood is False:
        ood_label = "In-distribution"
    else:
        ood_label = "-"
    threshold = result.get("decision_threshold")
    score = result.get("primary_score")
    score_text = "-" if score is None else f"{float(score):.4f}"
    threshold_text = "-" if threshold is None else f"{float(threshold):.4f}"
    error_text = str(result.get("error") or "").strip()
    view_consistency = dict(result.get("view_consistency", {}))
    uncertainty = dict(result.get("uncertainty_diagnostics", {}))
    robustness_warning_codes = [str(code) for code in list(view_consistency.get("warning_codes", []))]
    uncertainty_warning_codes = [str(code) for code in list(uncertainty.get("warning_codes", []))]
    if status == "error":
        robustness_status = "Prediction failed"
        robustness_accent = "#b91c1c"
    elif bool(view_consistency.get("stable")):
        robustness_status = "Stable across derived views"
        robustness_accent = "#0f766e"
    else:
        headline_warnings = robustness_warning_codes[:2] or uncertainty_warning_codes[:2]
        if headline_warnings:
            robustness_status = "Review recommended: " + ", ".join(headline_warnings)
        else:
            robustness_status = "Review recommended"
        robustness_accent = "#b45309"
    robustness_warning_text = ", ".join(robustness_warning_codes) if robustness_warning_codes else "-"
    uncertainty_warning_text = ", ".join(uncertainty_warning_codes) if uncertainty_warning_codes else "-"
    warning_details = _warning_details_html(view_consistency, uncertainty)
    visualization = dict(result.get("visualization", {}))
    visualization_line = ""
    if visualization.get("status") == "unavailable":
        visualization_line = (
            '<div style="color:#b45309;"><b style="color:#92400e;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('method') or '-'))} hazirlanamadi: "
            f"{escape(str(visualization.get('error') or '-'))}</div>"
        )
    elif visualization.get("method") == "occlusion_sensitivity":
        visualization_line = (
            '<div style="color:#374151;"><b style="color:#111827;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('view_name') or '-'))} gorunumu icin occlusion sensitivity haritasi hazir.</div>"
        )
    elif visualization.get("method") == "attention_map":
        visualization_line = (
            '<div style="color:#374151;"><b style="color:#111827;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('view_name') or '-'))} gorunumu icin attention map hazir.</div>"
        )
    return f"""
    <div style="border:1px solid #d0d7de;border-radius:10px;padding:16px;margin-top:12px;background:#ffffff;color:#111827;box-shadow:0 1px 3px rgba(15,23,42,0.08);">
      <div style="font-size:18px;font-weight:700;margin-bottom:8px;color:#111827;">{"Tahmin Basarisiz" if status == "error" else "Tahmin Sonucu"}</div>
      <div style="color:#374151;"><b style="color:#111827;">Adapter:</b> {escape(str(summary["resolved_adapter_dir"]))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Crop:</b> {escape(str(summary["crop_name"]))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Status:</b> {escape(status)}</div>
      <div style="color:#374151;"><b style="color:#111827;">Sinif:</b> {escape(str(result.get("predicted_class") or "-"))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Confidence:</b> {confidence_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Karari:</b> {ood_label}</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Score:</b> {score_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Karar Esigi:</b> {threshold_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Robustluk:</b> <span style="color:{robustness_accent};font-weight:600;">{escape(robustness_status)}</span></div>
      {visualization_line}
      <div style="margin-top:6px;color:#374151;"><b style="color:#111827;">Robustluk Aciklamasi:</b>{warning_details}</div>
      <div style="color:#374151;word-break:break-word;"><b style="color:#111827;">Goruntu:</b> {escape(str(image_path))}</div>
      {'<div style="color:#b91c1c;word-break:break-word;"><b style="color:#991b1b;">Hata:</b> ' + escape(error_text) + "</div>" if error_text else ""}
      <details style="margin-top:10px;color:#374151;">
        <summary style="cursor:pointer;color:#111827;font-weight:600;">Kompakt Teshis</summary>
        <div style="margin-top:8px;"><b style="color:#111827;">View warnings:</b> {escape(robustness_warning_text)}</div>
        <div><b style="color:#111827;">Uncertainty warnings:</b> {escape(uncertainty_warning_text)}</div>
      </details>
    </div>
    """


def _build_adapter_details_html(summary: dict[str, Any]) -> str:
    class_names = [str(name).strip() for name in list(summary.get("class_names", [])) if str(name).strip()]
    class_items = "".join(f'<li style="margin:4px 0;">{escape(name)}</li>' for name in class_names)
    class_block = (
        f'<ul style="margin:6px 0 0 18px;padding:0;color:#374151;">{class_items}</ul>'
        if class_items
        else '<div style="margin-top:6px;color:#6b7280;">Sinif listesi metadata\'da bulunamadi.</div>'
    )
    return f"""
    <div style="border:1px solid #d1d5db;border-radius:12px;padding:12px 14px;background:#fafafa;color:#111827;">
      <div style="font-size:15px;font-weight:700;margin-bottom:6px;">Adapter Bilgisi</div>
      <div style="color:#374151;"><b style="color:#111827;">Adapter:</b> {escape(str(summary.get("resolved_adapter_dir", "-")))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Crop:</b> {escape(str(summary.get("crop_name", "-")))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Part:</b> {escape(str(summary.get("part_name", "-")))}</div>
      <div style="margin-top:10px;font-weight:700;color:#111827;">Tanıyabileceği hastalıklar</div>
      {class_block}
    </div>
    """
