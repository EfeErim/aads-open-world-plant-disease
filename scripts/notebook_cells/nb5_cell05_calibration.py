# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 5.
# Keep notebook execute-only cells thin; edit behavior here.

if RUN_CALIBRATION:
    calibration_result = calibrate_router_surface(
        Path(ROUTER_EVAL_ROOT),
        config_env=CONFIG_ENV,
        device=DEVICE,
        preset=CALIBRATION_PRESET,
        sweep_specs=CUSTOM_SWEEPS,
        include_current=True,
        max_variants=MAX_VARIANTS,
        target_negative_false_accept_rate=TARGET_NEGATIVE_FALSE_ACCEPT_RATE,
        max_crop_accuracy_drop=MAX_CROP_ACCURACY_DROP,
        max_part_precision_drop=MAX_PART_PRECISION_DROP,
        max_part_recall_drop=MAX_PART_RECALL_DROP,
        max_wrong_part_rejection_drop=MAX_WRONG_PART_REJECTION_DROP,
        max_p95_latency_regression=MAX_P95_LATENCY_REGRESSION,
        include_samples=INCLUDE_SAMPLES_IN_OUTPUT,
        strategy=(CALIBRATION_STRATEGY or 'adaptive'),
        adaptive_top_k=12,
        adaptive_n_per_group=12,
        progress_every=PROGRESS_EVERY,
        collect_input_guard_scores=COLLECT_INPUT_GUARD_SCORES,
    )
    _write_json(CALIBRATION_OUTPUT, calibration_result)

    recommended = calibration_result.get('recommended', {})
    print('[CALIBRATION] recommended variant')
    print(json.dumps({
        'variant_id': recommended.get('variant_id'),
        'eligible': recommended.get('eligible'),
        'eligibility_reasons': recommended.get('eligibility_reasons'),
        'overrides': recommended.get('overrides'),
        'metrics': recommended.get('metrics'),
    }, indent=2))
else:
    print('[CALIBRATION] skipped')
