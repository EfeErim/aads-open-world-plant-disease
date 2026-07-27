import json
from pathlib import Path

from scripts.compare_m2_demo_results import compare_results, comparison_markdown, enrich_summary_manifest_sha256, main


def _summary(
    *,
    passed: int,
    failed: int,
    router: int,
    negative: int,
    opposite: int,
    targets: dict[str, tuple[int, int] | tuple[int, int, int]],
    total: int | None = None,
    manifest: str = "docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv",
    manifest_sha256: str | None = None,
):
    payload = {
        "created_at": "20260622T000000Z",
        "manifest": manifest,
        "summary": {
            "total": passed + failed if total is None else total,
            "passed": passed,
            "failed": failed,
            "failure_buckets": {"router": router},
            "per_target": {
                target: {
                    "total": values[0] + values[1] if len(values) == 2 else values[2],
                    "pass": values[0],
                    "fail": values[1],
                }
                for target, values in targets.items()
            },
        },
        "analysis_summary": {
            "negative_false_accepts": {"count": negative},
            "opposite_part_disease_labels": {"count": opposite},
            "prototype_correct_but_abstained": {"count": 10},
        },
    }
    if manifest_sha256 is not None:
        payload["manifest_sha256"] = manifest_sha256
    return payload


def test_compare_results_passes_when_candidate_improves_without_safety_regression():
    baseline = _summary(
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={
            "tomato__leaf": (61, 47),
            "tomato__fruit": (40, 23),
            "apricot__fruit": (35, 19),
            "strawberry__fruit": (30, 23),
        },
    )
    candidate = _summary(
        passed=370,
        failed=142,
        router=90,
        negative=5,
        opposite=10,
        targets={
            "tomato__leaf": (61, 47),
            "tomato__fruit": (45, 18),
            "apricot__fruit": (35, 19),
            "strawberry__fruit": (30, 23),
        },
    )

    comparison = compare_results(baseline=baseline, candidate=candidate)

    assert comparison["status"] == "pass"
    assert comparison["metrics"]["total_delta"] == 0
    assert comparison["metrics"]["failed_delta"] == -13
    assert comparison["checks"]["manifests_match"] is True
    assert comparison["checks"]["totals_match"] is True
    assert comparison["checks"]["focus_target_totals_match"] is True
    assert comparison["target_deltas"]["tomato__fruit"]["total_delta"] == 0
    assert comparison["target_deltas"]["tomato__fruit"]["pass_delta"] == 5


def test_compare_results_fails_when_negative_false_accepts_increase():
    baseline = _summary(
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
    )
    candidate = _summary(
        passed=370,
        failed=142,
        router=90,
        negative=7,
        opposite=10,
        targets={"grape__fruit": (35, 20)},
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))

    assert comparison["status"] == "fail"
    assert comparison["checks"]["negative_false_accepts_not_increased"] is False


def test_compare_results_fails_when_candidate_is_smaller_smoke_run():
    baseline = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
    )
    candidate = _summary(
        total=64,
        passed=60,
        failed=4,
        router=1,
        negative=0,
        opposite=0,
        targets={"grape__fruit": (35, 20)},
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))

    assert comparison["status"] == "fail"
    assert comparison["metrics"]["total_delta"] == -448
    assert comparison["checks"]["totals_match"] is False


def test_compare_results_fails_when_manifest_differs():
    baseline = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
        manifest="docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv",
    )
    candidate = _summary(
        total=512,
        passed=370,
        failed=142,
        router=90,
        negative=5,
        opposite=10,
        targets={"grape__fruit": (35, 20)},
        manifest="docs/demo_assets/m2_full_image_set/manifests/alternate_manifest.csv",
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))

    assert comparison["status"] == "fail"
    assert comparison["checks"]["manifests_match"] is False
    assert comparison["baseline_manifest"].endswith("m2_full_image_set_run_manifest.csv")
    assert comparison["candidate_manifest"].endswith("alternate_manifest.csv")


def test_compare_results_fails_when_manifest_hash_differs():
    baseline = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
        manifest_sha256="a" * 64,
    )
    candidate = _summary(
        total=512,
        passed=370,
        failed=142,
        router=90,
        negative=5,
        opposite=10,
        targets={"grape__fruit": (35, 20)},
        manifest_sha256="b" * 64,
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))

    assert comparison["status"] == "fail"
    assert comparison["checks"]["manifest_sha256_match"] is False
    assert comparison["baseline_manifest_sha256"] == "a" * 64
    assert comparison["candidate_manifest_sha256"] == "b" * 64


def test_enrich_summary_manifest_sha256_fills_existing_manifest_hash(tmp_path: Path):
    manifest_path = tmp_path / "manifests" / "m2.csv"
    manifest_path.parent.mkdir()
    manifest_path.write_text("image_id,source\nsample,image.jpg\n", encoding="utf-8")
    summary = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
        manifest="manifests/m2.csv",
    )

    enriched = enrich_summary_manifest_sha256(summary, repo_root=tmp_path)

    assert enriched is summary
    assert len(enriched["manifest_sha256"]) == 64
    assert set(enriched["manifest_sha256"]) != {""}
    assert enriched["manifest_sha256_source"] == "local_manifest_enriched"


def test_compare_results_warns_when_hash_matches_but_totals_differ():
    baseline = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
        manifest_sha256="a" * 64,
    )
    candidate = _summary(
        total=522,
        passed=393,
        failed=129,
        router=84,
        negative=1,
        opposite=8,
        targets={"grape__fruit": (42, 13)},
        manifest_sha256="a" * 64,
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))
    markdown = comparison_markdown(comparison)

    assert comparison["status"] == "fail"
    assert comparison["checks"]["manifest_sha256_match"] is True
    assert comparison["checks"]["totals_match"] is False
    assert "manifest_hash_matches_but_total_rows_differ" in comparison["warnings"]
    assert "- `manifest_hash_matches_but_total_rows_differ`" in markdown


def test_compare_results_fails_when_focus_target_total_changes():
    baseline = _summary(
        total=512,
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31, 55), "apricot__fruit": (35, 19, 54)},
    )
    candidate = _summary(
        total=512,
        passed=370,
        failed=142,
        router=90,
        negative=5,
        opposite=10,
        targets={"grape__fruit": (35, 20, 60), "apricot__fruit": (35, 19, 49)},
    )

    comparison = compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit", "apricot__fruit"))

    assert comparison["status"] == "fail"
    assert comparison["checks"]["focus_target_totals_match"] is False
    assert comparison["target_deltas"]["grape__fruit"]["total_delta"] == 5
    assert comparison["target_deltas"]["apricot__fruit"]["total_delta"] == -5


def test_comparison_markdown_includes_status_metrics_targets_and_checks():
    baseline = _summary(
        passed=357,
        failed=155,
        router=102,
        negative=6,
        opposite=15,
        targets={"grape__fruit": (24, 31)},
    )
    candidate = _summary(
        passed=370,
        failed=142,
        router=90,
        negative=5,
        opposite=10,
        targets={"grape__fruit": (35, 20)},
    )

    markdown = comparison_markdown(
        compare_results(baseline=baseline, candidate=candidate, targets=("grape__fruit",))
    )

    assert "# M2 Demo Result Comparison" in markdown
    assert "- Status: `pass`" in markdown
    assert "- Baseline manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`" in markdown
    assert "| `total_delta` | 0 |" in markdown
    assert "| `failed_delta` | -13 |" in markdown
    assert "| `grape__fruit` | 0 | 11 | -11 |" in markdown
    assert "| `manifests_match` | pass |" in markdown
    assert "| `manifest_sha256_match` | pass |" in markdown
    assert "| `totals_match` | pass |" in markdown
    assert "| `focus_target_totals_match` | pass |" in markdown
    assert "| `failed_not_increased` | pass |" in markdown


def test_main_returns_nonzero_on_regression(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(
            _summary(
                passed=357,
                failed=155,
                router=102,
                negative=6,
                opposite=15,
                targets={"grape__fruit": (24, 31)},
            )
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            _summary(
                passed=350,
                failed=162,
                router=120,
                negative=8,
                opposite=16,
                targets={"grape__fruit": (24, 31)},
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--focus-target",
            "grape__fruit",
            "--fail-on-regression",
        ]
    )

    assert exit_code == 1


def test_main_writes_json_and_markdown_outputs(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"
    baseline_path.write_text(
        json.dumps(
            _summary(
                passed=357,
                failed=155,
                router=102,
                negative=6,
                opposite=15,
                targets={"grape__fruit": (24, 31)},
            )
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            _summary(
                passed=370,
                failed=142,
                router=90,
                negative=5,
                opposite=10,
                targets={"grape__fruit": (35, 20)},
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--focus-target",
            "grape__fruit",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--fail-on-regression",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "pass"
    assert "| `grape__fruit` | 0 | 11 | -11 |" in markdown_path.read_text(encoding="utf-8")
