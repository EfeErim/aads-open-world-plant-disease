# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

holdout_recommended = (globals().get('holdout_result') or {}).get('recommended', {})
dev_recommended = (globals().get('calibration_result') or {})
recommended = {}
source = ''
if (globals().get('holdout_result') or {}).get('recommended', {}) and (globals().get('holdout_result') or {}).get('recommended', {}).get('eligible'):
    recommended = (globals().get('holdout_result') or {}).get('recommended')
    source = 'holdout'
elif dev_recommended.get('recommended', {}) and dev_recommended.get('recommended', {}).get('eligible'):
    recommended = dev_recommended.get('recommended')
    source = 'dev'
else:
    # if no eligible, prefer best_rejected from dev then holdout
    dev_best_rejected = dev_recommended.get('best_rejected') or {}
    holdout_best_rejected = (globals().get('holdout_result') or {}).get('best_rejected') or {}
    rejected = dev_best_rejected or holdout_best_rejected or {}

overrides = dict(recommended.get('overrides') or {})
if overrides:
    print(f'[CONFIG PREVIEW] Apply these {source}-validated eligible dotted-path values after review:')
    print(json.dumps(overrides, indent=2))
else:
    print('[CONFIG PREVIEW] No eligible override recommendation is available.')
    if rejected:
        print(json.dumps({
            'best_variant_id': rejected.get('variant_id'),
            'eligible': rejected.get('eligible'),
            'eligibility_reasons': rejected.get('eligibility_reasons'),
            'metrics': rejected.get('metrics'),
            'failure_analysis_summary': (dev_recommended.get('failure_analysis') or {}) if dev_recommended else {},
        }, indent=2))
