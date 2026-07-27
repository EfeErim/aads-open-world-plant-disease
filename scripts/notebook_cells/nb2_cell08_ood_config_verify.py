# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 8.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 5b: OOD Config Verify"):
    if "continual_config" not in STATE:
        raise RuntimeError("Once Cell 5 calistirilmali ki continual_config hazir olsun.")

    effective_params = dict(STATE.get("effective_params") or {})
    resolved_ood_cfg = dict(STATE["continual_config"].get("ood", {}))
    expected_ood_cfg = {
        "threshold_factor": float(effective_params["OOD_FACTOR"]),
        "sure_semantic_percentile": float(effective_params["SURE_SEMANTIC_PERCENTILE"]),
        "sure_confidence_percentile": float(effective_params["SURE_CONFIDENCE_PERCENTILE"]),
        "conformal_alpha": float(effective_params["CONFORMAL_ALPHA"]),
        "conformal_method": str(effective_params["CONFORMAL_METHOD"]),
        "conformal_raps_lambda": float(effective_params["CONFORMAL_RAPS_LAMBDA"]),
        "conformal_raps_k_reg": int(effective_params["CONFORMAL_RAPS_K_REG"]),
    }

    print("[VERIFY][OOD][expected]", json.dumps(expected_ood_cfg, sort_keys=True))
    print("[VERIFY][OOD][resolved]", json.dumps({k: resolved_ood_cfg.get(k) for k in expected_ood_cfg}, sort_keys=True))

    mismatches = []
    for key, expected in expected_ood_cfg.items():
        actual = resolved_ood_cfg.get(key)
        if isinstance(expected, float):
            try:
                actual_float = float(actual)
            except Exception:
                mismatches.append(f"{key}: expected={expected} actual={actual}")
                continue
            if abs(actual_float - expected) > 1e-12:
                mismatches.append(f"{key}: expected={expected} actual={actual_float}")
        elif actual != expected:
            mismatches.append(f"{key}: expected={expected} actual={actual}")

    if mismatches:
        raise RuntimeError("OOD config mismatch:\n - " + "\n - ".join(mismatches))

    print("[VERIFY][OOD] OK: cozulmus OOD ayari istenen parametrelerle eslesiyor.")