# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 11.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 8: Adapter Save"):
    rt("Cell 8: adapter save started", phase="export")

    if STATE.get("adapter") is None:
        raise RuntimeError("Once Cell 5 calistirilmali.")

    crop_name = str(STATE.get("crop_name") or CROP_NAME).strip().lower()
    part_name = str(STATE.get("part_name") or PART_NAME or "unspecified").strip().lower() or "unspecified"
    checkpoint_dir = build_adapter_bundle_root(ROOT / "outputs" / "colab_notebook_training", crop_name, part_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    STATE["adapter"].save_adapter(str(checkpoint_dir))
    asset_dir = checkpoint_dir / "continual_sd_lora_adapter"
    STATE["adapter_export_dir"] = asset_dir

    print("Kaydedilen adapter klasoru:", asset_dir)
    if not asset_dir.exists():
        raise RuntimeError(f"Beklenen adapter klasoru bulunamadi: {asset_dir}")

    telemetry_adapter_root = TELEMETRY.artifacts_dir / "adapter_export" / crop_name / part_name / "continual_sd_lora_adapter"
    for path_in_adapter in sorted(asset_dir.rglob("*")):
        if path_in_adapter.is_file():
            relative_path = path_in_adapter.relative_to(LOCAL_OUTPUT_DIR).as_posix()
            TELEMETRY.copy_artifact_file(path_in_adapter, f"adapter_export/{relative_path}")
            print(" -", path_in_adapter.relative_to(ROOT))

    print("Telemetry adapter klasoru:", telemetry_adapter_root)
    TELEMETRY.update_latest(
        {
            "phase": "adapter_saved",
            "adapter_dir": str(telemetry_adapter_root),
        }
    )
    rt("Cell 8: adapter save completed", phase="export")
