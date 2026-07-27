# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 4.
# Keep notebook execute-only cells thin; edit behavior here.

if RUN_BASELINE_EVAL:
    baseline_result = evaluate_router_surface(
        Path(ROUTER_EVAL_ROOT),
        config_env=CONFIG_ENV,
        device=DEVICE,
    )
    _write_json(BASELINE_EVAL_OUTPUT, baseline_result)
    metrics = baseline_result.get('metrics', {})
    print('[BASELINE] metrics')
    for key in (
        'sample_count',
        'crop_accuracy',
        'negative_false_accept_rate',
        'abstention_rate',
        'part_non_unknown_precision',
        'part_recall',
        'wrong_part_rejection_rate',
        'mean_latency_ms',
        'p95_latency_ms',
    ):
        print(f'  {key}: {metrics.get(key)}')
else:
    print('[BASELINE] skipped')
