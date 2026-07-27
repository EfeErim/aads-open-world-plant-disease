# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

if RUN_HOLDOUT_VALIDATION:
    ranked_variants = list((globals().get('calibration_result') or {}).get('variants') or [])
    candidates = [
        dict(row.get('overrides') or {})
        for row in ranked_variants
        if row.get('variant_id') != 'baseline' and row.get('eligible') and row.get('overrides')
    ][: int(HOLDOUT_TOP_K)]

    if not candidates:
        holdout_result = {}
        print('[HOLDOUT] skipped: no eligible dev-set candidate overrides are available.')
    else:
        holdout_result = validate_router_candidate_overrides(
            Path(HOLDOUT_EVAL_ROOT),
            candidate_overrides=candidates,
            config_env=CONFIG_ENV,
            device=DEVICE,
            target_negative_false_accept_rate=TARGET_NEGATIVE_FALSE_ACCEPT_RATE,
            max_crop_accuracy_drop=MAX_CROP_ACCURACY_DROP,
            max_part_precision_drop=MAX_PART_PRECISION_DROP,
            max_part_recall_drop=MAX_PART_RECALL_DROP,
            max_wrong_part_rejection_drop=MAX_WRONG_PART_REJECTION_DROP,
            max_p95_latency_regression=MAX_P95_LATENCY_REGRESSION,
            include_samples=INCLUDE_SAMPLES_IN_OUTPUT,
            strategy=CALIBRATION_STRATEGY,
            progress_every=PROGRESS_EVERY,
            collect_input_guard_scores=COLLECT_INPUT_GUARD_SCORES,
        )
        _write_json(HOLDOUT_VALIDATION_OUTPUT, holdout_result)

        print('[HOLDOUT] accepted candidate')
        print(json.dumps({
            'variant_id': (holdout_result.get('recommended') or {}).get('variant_id'),
            'eligible': (holdout_result.get('recommended') or {}).get('eligible'),
            'eligibility_reasons': (holdout_result.get('recommended') or {}).get('eligibility_reasons'),
            'overrides': (holdout_result.get('recommended') or {}).get('overrides'),
            'metrics': (holdout_result.get('recommended') or {}).get('metrics'),
        }, indent=2))
else:
    print('[HOLDOUT] skipped')
