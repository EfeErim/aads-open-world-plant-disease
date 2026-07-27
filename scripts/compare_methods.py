"""Compare OOD methods (ensemble/knn/energy) using production_readiness.json method-level pooled metrics.

Outputs: .runtime_tmp/knn_vs_ensemble_summary.csv
"""
import json
from pathlib import Path
import csv

RUNTIME = Path('.runtime_tmp')
RUNTIME.mkdir(parents=True, exist_ok=True)
OUT = RUNTIME / 'knn_vs_ensemble_summary.csv'

# target runs (from user's list)
TARGET_RUNS = [
    ('grape','fruit','grape_fruit_2026-05-24_14-47-58'),
    ('grape','leaf','grape_leaf_2026-05-24_14-41-21'),
    ('apricot','leaf','apricot_leaf_2026-05-24_14-15-18'),
    ('tomato','fruit','tomato_fruit_2026-05-24_15-06-48')
]

rows = []
for crop, part, run in TARGET_RUNS:
    # locate production_readiness.json by searching under run dir
    prod = None
    run_dir = Path('runs') / crop / part / run
    # prefer outputs/colab_notebook_training/artifacts/production_readiness.json if present
    p_outputs = Path('runs')/crop/part/run/'outputs'/'colab_notebook_training'/'artifacts'/'production_readiness.json'
    p_outputs2 = Path('runs')/crop/part/run/'outputs'/'colab_notebook_training'/'artifacts'/'test'/'production_readiness.json'
    p_telemetry = Path('runs')/crop/part/run/'telemetry'/'artifacts'/'production_readiness.json'
    prod = None
    if p_outputs.exists():
        prod = p_outputs
    elif p_outputs2.exists():
        prod = p_outputs2
    elif p_telemetry.exists():
        prod = p_telemetry
    else:
        if run_dir.exists():
            for p in run_dir.rglob('production_readiness.json'):
                prod = p
                break
    # try alternative path where crop_part combined
    if not prod:
        run_dir2 = Path('runs') / f"{crop}_{part}" / run
        if run_dir2.exists():
            for p in run_dir2.rglob('production_readiness.json'):
                prod = p
                break
    if not prod:
        print('production_readiness.json not found for', run)
        continue
    with open(prod, 'r', encoding='utf-8') as f:
        pr = json.load(f)
    # ood_method_comparison may be nested under context; search recursively
    def find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                res = find_key(v, key)
                if res is not None:
                    return res
        return None
    omc = find_key(pr, 'ood_method_comparison') or {}
    methods = omc.get('methods', {})
    for m, info in methods.items():
        pooled = info.get('pooled_metrics', {})
        rows.append({
            'run': run,
            'crop': crop,
            'part': part,
            'method': m,
            'ood_auroc': pooled.get('ood_auroc'),
            'ood_fpr': pooled.get('ood_false_positive_rate'),
            'ood_samples': pooled.get('ood_samples'),
            'in_distribution_samples': pooled.get('in_distribution_samples')
        })

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['run','crop','part','method','ood_auroc','ood_fpr','ood_samples','in_distribution_samples'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote', OUT)
