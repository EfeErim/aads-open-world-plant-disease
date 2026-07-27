"""Evaluate OOD methods per run using production_readiness.json per-slice metrics.

Outputs:
 - .runtime_tmp/method_selection_per_slice.csv  (per-run per-slice per-method metrics)
 - .runtime_tmp/method_selection_recommendations.csv (per-run recommended method and rationale)

Usage: python scripts/evaluate_methods_per_slice.py
"""
import csv
import json
from pathlib import Path

RUNTIME = Path('.runtime_tmp')
RUNTIME.mkdir(parents=True, exist_ok=True)
OUT_SLICES = RUNTIME / 'method_selection_per_slice.csv'
OUT_RECS = RUNTIME / 'method_selection_recommendations.csv'


def find_pr_for_run(crop, part, run):
    run_dir = Path('runs') / crop / part / run
    candidates = [
        run_dir / 'outputs' / 'colab_notebook_training' / 'artifacts' / 'production_readiness.json',
        run_dir / 'outputs' / 'colab_notebook_training' / 'artifacts' / 'test' / 'production_readiness.json',
        run_dir / 'telemetry' / 'artifacts' / 'production_readiness.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    if run_dir.exists():
        for p in run_dir.rglob('production_readiness.json'):
            return p
    # try alternative folder style
    run_dir2 = Path('runs') / f"{crop}_{part}" / run
    if run_dir2.exists():
        for p in run_dir2.rglob('production_readiness.json'):
            return p
    return None


def load_targets_from_delta(delta_csv):
    targets = []
    if not delta_csv.exists():
        return targets
    with open(delta_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            cur = r.get('current_run')
            cp = r.get('crop_part')
            if not cur:
                continue
            # infer crop, part from crop_part
            if cp and '_' in cp:
                crop, part = cp.split('_', 1)
            else:
                parts = cur.split('_')
                crop = parts[0]
                part = parts[1] if len(parts) > 1 else ''
            targets.append((crop, part, cur))
    return targets


def choose_best_method_for_slice(method_metrics):
    # choose method with minimal ood_false_positive_rate; tie-breaker: maximal ood_auroc
    best = None
    for m, mm in method_metrics.items():
        fpr = mm.get('ood_false_positive_rate')
        auroc = mm.get('ood_auroc')
        if fpr is None:
            continue
        if best is None:
            best = (m, fpr, auroc)
            continue
        if fpr < best[1] - 1e-12:
            best = (m, fpr, auroc)
        elif abs(fpr - best[1]) < 1e-12 and auroc is not None and best[2] is not None and auroc > best[2]:
            best = (m, fpr, auroc)
    return best[0] if best else None


def main():
    delta = Path('.runtime_tmp') / 'runs_delta_report.csv'
    targets = load_targets_from_delta(delta)
    # fallback: if delta empty, use a small built-in list (keeps parity with other scripts)
    if not targets:
        targets = [
            ('grape','fruit','grape_fruit_2026-05-24_14-47-58'),
            ('grape','leaf','grape_leaf_2026-05-24_14-41-21'),
            ('apricot','leaf','apricot_leaf_2026-05-24_14-15-18'),
            ('tomato','fruit','tomato_fruit_2026-05-24_15-06-48')
        ]

    slice_rows = []
    rec_rows = []

    for crop, part, run in targets:
        print('Evaluating', run)
        pr = find_pr_for_run(crop, part, run)
        if not pr:
            print('  production_readiness.json not found for', run)
            continue
        try:
            with open(pr, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:
            print('  failed to load', pr, e)
            continue

        # collect per-slice method metrics
        ood_types = doc.get('context', {}).get('ood_type_breakdown') or {}
        # method comparison might also contain method-level pooled/worst_slice
        method_comp = doc.get('context', {}).get('ood_method_comparison') or doc.get('ood_method_comparison') or {}

        # for each slice, extract method_metrics if present
        method_wins = {}
        method_weight = {}
        for slice_name, slice_info in ood_types.items():
            mm = slice_info.get('method_metrics') or {}
            sample_count = slice_info.get('sample_count') or slice_info.get('metrics', {}).get('ood_samples') or 0
            for m, metrics in mm.items():
                slice_rows.append({
                    'run': run,
                    'crop': crop,
                    'part': part,
                    'slice': slice_name,
                    'method': m,
                    'ood_auroc': metrics.get('ood_auroc'),
                    'ood_false_positive_rate': metrics.get('ood_false_positive_rate'),
                    'ood_samples': metrics.get('ood_samples') or sample_count,
                    'in_distribution_samples': metrics.get('in_distribution_samples')
                })
            best = choose_best_method_for_slice(mm)
            if best:
                method_wins[best] = method_wins.get(best, 0) + 1
                method_weight[best] = method_weight.get(best, 0) + (sample_count or 0)

        # also include pooled metrics per method (if available)
        methods = method_comp.get('methods') or {}
        for m, info in methods.items():
            pooled = info.get('pooled_metrics') or {}
            slice_rows.append({
                'run': run,
                'crop': crop,
                'part': part,
                'slice': 'POOLED',
                'method': m,
                'ood_auroc': pooled.get('ood_auroc'),
                'ood_false_positive_rate': pooled.get('ood_false_positive_rate'),
                'ood_samples': pooled.get('ood_samples'),
                'in_distribution_samples': pooled.get('in_distribution_samples')
            })

        # pick recommendation: method with highest weighted sample wins, fallback to pooled best fpr
        recommended = None
        rationale = ''
        if method_weight:
            # choose method with highest weighted sample_count wins
            recommended = max(method_weight.items(), key=lambda x: x[1])[0]
            rationale = f"slice-weighted wins (by sample_count): {method_weight}"
        else:
            # fallback: choose method with lowest pooled fpr
            best_pool = None
            for m, info in methods.items():
                p = info.get('pooled_metrics', {})
                fpr = p.get('ood_false_positive_rate')
                auroc = p.get('ood_auroc')
                if fpr is None:
                    continue
                if best_pool is None or fpr < best_pool[1] or (abs(fpr - best_pool[1]) < 1e-12 and auroc and best_pool[2] and auroc > best_pool[2]):
                    best_pool = (m, fpr, auroc)
            if best_pool:
                recommended = best_pool[0]
                rationale = f"pooled best fpr: {best_pool[1]}"

        rec_rows.append({'run': run, 'crop': crop, 'part': part, 'recommended_method': recommended or '', 'rationale': rationale})

    # write per-slice CSV
    with open(OUT_SLICES, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run','crop','part','slice','method','ood_auroc','ood_false_positive_rate','ood_samples','in_distribution_samples']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in slice_rows:
            w.writerow(r)

    with open(OUT_RECS, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run','crop','part','recommended_method','rationale']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rec_rows:
            w.writerow(r)

    print('Wrote', OUT_SLICES, 'and', OUT_RECS)


if __name__ == '__main__':
    main()
