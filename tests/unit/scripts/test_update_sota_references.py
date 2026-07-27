from scripts.update_sota_references import (
    CANDIDATE_SECTION_BEGIN,
    CANDIDATE_SECTION_END,
    infer_repo_action_hint,
    is_relevant_candidate,
    render_candidate_section,
    render_repo_opportunity_scan,
    scan_repo_opportunities,
    should_preserve_existing_candidate_section,
    update_guide_with_candidate_section,
)


def test_update_guide_replaces_existing_candidate_section():
    old_section = "\n".join(
        [
            CANDIDATE_SECTION_BEGIN,
            "#### Latest Automated Candidate Scan",
            "",
            "Generated: `old`",
            "",
            CANDIDATE_SECTION_END,
        ]
    )
    guide = f"# Guide\n\n{old_section}\n\n### Phase 2 Checklist"
    new_section = render_candidate_section([], "2026-05-13T18:00:00Z")

    updated = update_guide_with_candidate_section(guide, new_section)

    assert updated.count(CANDIDATE_SECTION_BEGIN) == 1
    assert updated.count(CANDIDATE_SECTION_END) == 1
    assert "Generated: `old`" not in updated
    assert "Generated: `2026-05-13T18:00:00Z`" in updated
    assert "### Phase 2 Checklist" in updated


def test_update_guide_replaces_section_when_candidate_text_contains_backslashes():
    old_section = "\n".join([CANDIDATE_SECTION_BEGIN, "old", CANDIDATE_SECTION_END])
    guide = f"# Guide\n\n{old_section}\n"
    new_section = "\n".join([CANDIDATE_SECTION_BEGIN, r"uses \s and \alpha", CANDIDATE_SECTION_END])

    updated = update_guide_with_candidate_section(guide, new_section)

    assert r"uses \s and \alpha" in updated


def test_update_guide_inserts_candidate_section_before_phase_2_when_missing():
    guide = "# Guide\n\n### Phase 1 Checklist\n\n### Phase 2 Checklist\n"
    section = render_candidate_section([], "2026-05-13T18:00:00Z")

    updated = update_guide_with_candidate_section(guide, section)

    assert updated.index(CANDIDATE_SECTION_BEGIN) < updated.index("### Phase 2 Checklist")
    assert "No new papers found for configured queries." in updated


def test_render_candidate_section_reports_query_errors():
    section = render_candidate_section(
        [],
        "2026-05-13T18:00:00Z",
        query_errors=[("bioclip", "network unavailable")],
    )

    assert "Candidate scan could not query all configured sources:" in section
    assert "`bioclip`: network unavailable" in section
    assert "No new papers found for configured queries." not in section


def test_render_candidate_section_includes_repo_action_hint():
    section = render_candidate_section(
        [
            {
                "query": "mahalanobis ood",
                "title": "Dual Feature Decoupling for Fine-Grained OOD Detection",
                "summary": "Fine-grained out-of-distribution detection with background factors.",
                "published": "2026-06-04T00:38:21Z",
                "authors": ["Example Author"],
                "link": "https://arxiv.org/abs/2606.05536",
            }
        ],
        "2026-06-07T13:46:33Z",
    )

    assert "- Repo action hint:" in section
    assert "near-OOD slices" in section


def test_infer_repo_action_hint_maps_plant_ssl_to_augmentation_review():
    hint = infer_repo_action_hint(
        {
            "query": "bioclip",
            "title": "Self-Supervised Learning of Plant Image Representations",
            "summary": "Plant SSL shows Gaussian blur and grayscale can erase fine-grained cues.",
        }
    )

    assert "augmentation policy" in hint
    assert "subtle leaf/symptom cues" in hint


def test_render_candidate_section_summarizes_permission_errors():
    section = render_candidate_section(
        [],
        "2026-05-13T18:00:00Z",
        query_errors=[("bioclip", "Failed to connect: [WinError 10013] permission denied")],
    )

    assert "`bioclip`: network access blocked by local permissions" in section
    assert "WinError" not in section


def test_all_query_failures_preserve_existing_candidate_section():
    assert should_preserve_existing_candidate_section(0, [("bioclip", "query timed out")])
    assert not should_preserve_existing_candidate_section(1, [("bioclip", "query timed out")])
    assert not should_preserve_existing_candidate_section(0, [])


def test_candidate_section_can_embed_repo_opportunity_scan():
    section = render_candidate_section(
        [],
        "2026-05-13T18:00:00Z",
        extra_sections=[render_repo_opportunity_scan([{"kind": "bug", "file": "scripts/x.py", "line": 7, "summary": "FIXME marker"}])],
    )

    assert section.index("#### Repo Bug / Weak Point / Suboptimal Code Scan") < section.index(CANDIDATE_SECTION_END)
    assert "`bug` [scripts/x.py:7]: FIXME marker" in section


def test_scan_repo_opportunities_finds_markers_and_canonical_guide_violations(tmp_path):
    workflow = tmp_path / ".github" / "workflows" / "sota_auto_update.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("python scripts/update_sota_references.py --output docs/SOTA_AUTOMATION_UPDATES.md\n", encoding="utf-8")
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text("# TODO: tighten validation\n", encoding="utf-8")

    opportunities = scan_repo_opportunities(tmp_path, max_items=5)
    marker_opportunities = scan_repo_opportunities(tmp_path, max_items=5, roots=("scripts",))

    assert any(item["kind"] == "bug" and item["file"] == ".github/workflows/sota_auto_update.yml" for item in opportunities)
    assert any(item["kind"] == "improvement" and item["file"] == "scripts/example.py" for item in marker_opportunities)


def test_scan_repo_opportunities_ignores_plain_bug_words_without_marker_format(tmp_path):
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text('text = "Repo Bug / Weak Point / Suboptimal Code Scan"\n', encoding="utf-8")

    opportunities = scan_repo_opportunities(tmp_path, max_items=5, roots=("scripts",))

    assert opportunities == []


def test_scan_repo_opportunities_ignores_markers_inside_python_strings(tmp_path):
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text('text = "# TODO: fixture text, not a code marker"\n', encoding="utf-8")

    opportunities = scan_repo_opportunities(tmp_path, max_items=5, roots=("scripts",))

    assert opportunities == []


def test_scan_repo_opportunities_flags_src_importing_scripts(tmp_path):
    module = tmp_path / "src" / "package" / "runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("from scripts.audit_code_organization import main\n", encoding="utf-8")

    opportunities = scan_repo_opportunities(tmp_path, max_items=5, roots=("src",))

    assert opportunities == [
        {
            "kind": "bug",
            "file": "src/package/runtime.py",
            "line": 1,
            "summary": "src imports scripts; durable logic must not depend on orchestration surfaces",
        }
    ]


def test_scan_repo_opportunities_flags_swallowed_broad_exception(tmp_path):
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text("try:\n    risky()\nexcept Exception: pass\n", encoding="utf-8")

    opportunities = scan_repo_opportunities(tmp_path, max_items=5, roots=("scripts",))

    assert opportunities == [
        {
            "kind": "weak_point",
            "file": "scripts/example.py",
            "line": 3,
            "summary": "broad exception is swallowed with pass; preserve error context or narrow the exception",
        }
    ]


def test_is_relevant_candidate_filters_broad_arxiv_noise():
    assert is_relevant_candidate(
        {
            "query": "router calibration",
            "title": "Calibrated router handoff under distribution shift",
            "summary": "Risk-coverage curves for selective prediction.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "out-of-distribution detection",
            "title": "Unveiling hidden Lyman alpha emitters",
            "summary": "A good Dark Energy spectroscopic survey model.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "router calibration",
            "title": "Stable layers for image decomposition",
            "summary": "A calibration step improves reinforcement learning rewards.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "conformal prediction",
            "title": "Quiver approach to symmetry theories",
            "summary": "This work studies conformal field theories.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "mahalanobis ood",
            "title": "OOD-GraphLLM for drug synergy prediction",
            "summary": "Out-of-distribution generalization for molecular graphs.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "mahalanobis ood",
            "title": "Physics-Aware Auxiliary Losses Improve Out-of-Distribution Generalization of a GNN Synthesizability Filter",
            "summary": "Auxiliary supervision improves molecular graph generalization.",
        }
    )
    assert is_relevant_candidate(
        {
            "query": "logitnorm",
            "title": "Enhancing Out-of-Distribution Detection with Extended Logit Normalization",
            "summary": "The method improves image classification under distribution shift.",
        }
    )
    assert is_relevant_candidate(
        {
            "query": "bioclip",
            "title": "Self-Supervised Learning of Plant Image Representations",
            "summary": "Plant recognition models outperform BioCLIP on fine-grained leaf images.",
        }
    )
    assert not is_relevant_candidate(
        {
            "query": "bioclip",
            "title": "PRIMA: Boosting Animal Mesh Recovery with Biological Priors",
            "summary": "BioCLIP embeddings improve quadruped mesh reconstruction.",
        }
    )
