# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 10.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 7: OOD Calibration"):
    from src.training.notebook_runtime_helpers import calibrate_notebook_ood_policy

    if STATE.get("adapter") is None or STATE.get("loaders") is None:
        raise RuntimeError("Once engine init hucresini calistirin.")

    adapter = STATE["adapter"]
    calibration_result = calibrate_notebook_ood_policy(
        adapter=adapter,
        loaders=STATE["loaders"],
        continual_config=dict(STATE.get("continual_config") or {}),
    )
    calibration = calibration_result["calibration"]
    policy = calibration_result["policy"]
    STATE["calibration"] = calibration
    STATE["ood_policy_selection"] = policy

    num_classes = calibration.get("ood_calibration", {}).get("num_classes", 0)
    version = calibration.get("ood_calibration", {}).get("version", 0)
    print(
        f"[OOD] Kalibrasyon tamamlandi. classes={num_classes} version={version} "
        f"method={policy['selected_primary_score_method']} source={policy['selection_source']}"
    )
    TELEMETRY.update_latest(
        {
            "phase": "ood_calibrated",
            "ood_num_classes": num_classes,
            "ood_version": version,
            "ood_requested_primary_score_method": policy["requested_primary_score_method"],
            "ood_primary_score_method": policy["selected_primary_score_method"],
            "ood_primary_score_selection_source": policy["selection_source"],
        }
    )
