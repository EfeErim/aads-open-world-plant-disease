# Auto-extracted from colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 3b: Guncelleme ve Erisim Kontrolu"):
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
    from scripts.notebook_helpers.nb0_grouped_dataset_prep_helpers import DatasetReleaseAccessBlocker
    from src.pipeline.adapter_release import resolve_token

    access_report = collect_notebook_access_report(
        repo_root=ROOT,
        hf_model_ids=[PREP_DINOV3_MODEL_ID, PREP_BIOCLIP_MODEL_ID],
    )
    STATE["access_report"] = access_report
    print_notebook_access_report(access_report, print_fn=print)
    dataset_release_read_token = resolve_token(write=False)
    if DATASET_RELEASE_TAG and not dataset_release_read_token:
        raise DatasetReleaseAccessBlocker(
            "DATASET_RELEASE_ACCESS_BLOCKER: AADS_GITHUB_RELEASE_READ_TOKEN is required for the private release"
        )
    STATE["dataset_release_access"] = {
        "required": bool(DATASET_RELEASE_TAG),
        "read_token_available": bool(dataset_release_read_token),
        "write_token_requested": False,
    }
    print(
        "[DATASET RELEASE] "
        f"read_access_ready={bool(dataset_release_read_token)} write_token_requested=False"
    )
    TELEMETRY.update_latest(
        {
            "phase": "access_checked",
            "github_read_access": access_report.get("github", {}).get("read_access_mode"),
            "repo_update_relation": access_report.get("repo_updates", {}).get("relation"),
            "hf_access_mode": access_report.get("huggingface", {}).get("access_mode"),
        }
    )
