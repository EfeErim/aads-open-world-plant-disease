from __future__ import annotations

from src.ood.recovery import TARGET_ADAPTERS
from src.ood.score_comparison import build_score_comparison


def _rows() -> list[dict]:
    rows = []
    for target in TARGET_ADAPTERS:
        for index, label in enumerate([0, 0, 0, 0, 1, 1, 1, 1]):
            rows.append(
                {
                    "target": target,
                    "split_role": "ood_dev",
                    "ood_label": label,
                    "ood_type": "same_crop_unsupported_disease" if label else "",
                    "classification_correct": True if not label else None,
                    "scores": {
                        "ensemble": 0.1 + 0.7 * label + index * 0.001,
                        "energy": 0.2 + 0.6 * label + index * 0.001,
                        "knn": 0.3 + 0.5 * label + index * 0.001,
                    },
                }
            )
    return rows


def test_comparison_selects_all_targets_from_dev_only() -> None:
    report = build_score_comparison(_rows(), target_fpr=0.05)

    assert report["ok"] is True
    assert report["selected_target_count"] == 8
    assert report["final_ood_test_used_for_selection"] is False
    assert all(adapter["selected_method"] in {"ensemble", "energy", "knn"} for adapter in report["adapters"])
    assert all(
        adapter["methods"][adapter["selected_method"]]["threshold_semantics"] == "is_ood = score > threshold"
        for adapter in report["adapters"]
    )


def test_final_test_rows_are_rejected_for_selection() -> None:
    rows = _rows()
    rows[0]["split_role"] = "ood_test"

    report = build_score_comparison(rows)

    assert report["ok"] is False
    assert any(issue["code"] == "non_dev_selection_evidence" for issue in report["issues"])


def test_missing_score_method_is_rejected() -> None:
    rows = _rows()
    del rows[0]["scores"]["knn"]

    report = build_score_comparison(rows)

    assert report["ok"] is False
    assert any(issue["code"] == "score_method_missing" for issue in report["issues"])
