# Auto-extracted from colab_notebooks/1_identify_crop_part_with_router.ipynb cell 7.
# Keep notebook execute-only cells thin; edit behavior here.

import json
from pathlib import Path

if RUN_ROUTER_EVAL and ROUTER_EVAL_ROOT:
    from scripts.evaluate_router_part_surface import evaluate_router_part_surface, _parse_grid

    eval_result = evaluate_router_part_surface(
        Path(ROUTER_EVAL_ROOT),
        config_env=CONFIG_ENV,
        device=DEVICE,
        min_confidence_grid=_parse_grid(EVAL_MIN_CONFIDENCE_GRID),
        margin_grid=_parse_grid(EVAL_MARGIN_GRID),
    )
    if ROUTER_EVAL_OUTPUT:
        output_path = Path(ROUTER_EVAL_OUTPUT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(eval_result, indent=2), encoding='utf-8')
        print(f'[EVAL] wrote {output_path}')

    print(f"[EVAL] dataset_root={eval_result['dataset_root']}")
    print(f"[EVAL] sample_count={eval_result['sample_count']} crop_accuracy={eval_result['crop_accuracy']:.4f}")
    for crop_name, crop_summary in sorted((eval_result.get('crops') or {}).items()):
        print(
            f"[EVAL] crop={crop_name} sample_count={crop_summary.get('sample_count', 0)} "
            f"crop_accuracy={float(crop_summary.get('crop_accuracy', 0.0) or 0.0):.4f} "
            f"non_unknown_precision={float(crop_summary.get('non_unknown_precision', 0.0) or 0.0):.4f} "
            f"abstention_rate={float(crop_summary.get('abstention_rate', 0.0) or 0.0):.4f} "
            f"unsupported_part_emission={int(crop_summary.get('unsupported_part_emission', 0) or 0)}"
        )
    threshold_sweep = dict(eval_result.get('threshold_sweep') or {})
    recommended = dict(threshold_sweep.get('recommended') or {})
    if recommended:
        print(
            f"[EVAL] recommended min_confidence={float(recommended.get('min_confidence', 0.0) or 0.0):.3f} "
            f"margin={float(recommended.get('margin', 0.0) or 0.0):.3f} "
            f"precision={float(recommended.get('non_unknown_precision', 0.0) or 0.0):.4f} "
            f"abstention={float(recommended.get('abstention_rate', 0.0) or 0.0):.4f}"
        )
else:
    print('[EVAL] RUN_ROUTER_EVAL kapali veya ROUTER_EVAL_ROOT ayarlanmadi.')
