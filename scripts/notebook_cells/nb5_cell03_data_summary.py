# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 3.
# Keep notebook execute-only cells thin; edit behavior here.

eval_root = Path(ROUTER_EVAL_ROOT)
samples = discover_eval_samples(eval_root) if eval_root.exists() else []
print(f'[DATA] root={eval_root} exists={eval_root.exists()} samples={len(samples)}')

if not samples:
    print('[DATA] No router eval images found. Create the data/router_eval layout before running calibration.')
else:
    counts = {}
    for sample in samples:
        group = str(sample.get('group', 'unknown'))
        counts[group] = counts.get(group, 0) + 1
    for group, count in sorted(counts.items()):
        print(f'[DATA] {group}: {count}')
