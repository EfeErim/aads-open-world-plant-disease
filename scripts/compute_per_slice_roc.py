"""Compute per-slice per-method ROC and FPR@TPR95 from predictions.csv files.

Outputs:
 - .runtime_tmp/per_slice_sample_roc.csv

Usage: python scripts/compute_per_slice_roc.py
"""
import csv
import json
from pathlib import Path
import math

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    roc_auc_score = None
    roc_curve = None

RUNTIME = Path('.runtime_tmp')
RUNTIME.mkdir(parents=True, exist_ok=True)
OUT = RUNTIME / 'per_slice_sample_roc.csv'

METHODS = ['ensemble', 'knn', 'energy']


def find_predictions(crop, part, run):
    run_dir = Path('runs') / crop / part / run
    candidates = [
        run_dir / 'outputs' / 'colab_notebook_training' / 'artifacts' / 'test' / 'predictions.csv',
        run_dir / 'outputs' / 'colab_notebook_training' / 'artifacts' / 'predictions.csv',
        run_dir / 'telemetry' / 'artifacts' / 'test' / 'predictions.csv',
        run_dir / 'telemetry' / 'artifacts' / 'predictions.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    if run_dir.exists():
        for p in run_dir.rglob('predictions.csv'):
            return p
    # try alternative folder style
    run_dir2 = Path('runs') / f"{crop}_{part}" / run
    if run_dir2.exists():
        for p in run_dir2.rglob('predictions.csv'):
            return p
    return None


def find_pr(crop, part, run):
    run_dir = Path('runs') / crop / part / run
    for p in run_dir.rglob('production_readiness.json'):
        return p
    return None


def detect_score_columns(header, method):
    # return column name matching method score if present
    candidates = []
    low = [h.lower() for h in header]
    for i, h in enumerate(header):
        lh = h.lower()
        if method in lh and 'score' in lh:
            candidates.append(h)
    # more general patterns
    for h in header:
        lh = h.lower()
        if ('ood' in lh or 'score' in lh) and method in lh:
            if h not in candidates:
                candidates.append(h)
    return candidates[0] if candidates else None


def compute_fpr_at_tpr95(y_true, scores):
    if roc_curve is None:
        return None
    try:
        fpr, tpr, thr = roc_curve(y_true, scores)
    except Exception:
        return None
    # find minimum fpr where tpr >= 0.95
    target = 0.95
    for fp, tp in zip(fpr, tpr):
        if tp >= target:
            return float(fp)
    return float(fpr[-1])


def compute_auroc(y_true, scores):
    if roc_auc_score is None:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def main():
    delta = Path('.runtime_tmp') / 'runs_delta_report.csv'
    targets = []
    if delta.exists():
        with open(delta, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                cur = r.get('current_run')
                cp = r.get('crop_part')
                if not cur:
                    continue
                if cp and '_' in cp:
                    crop, part = cp.split('_', 1)
                else:
                    parts = cur.split('_')
                    crop = parts[0]
                    part = parts[1] if len(parts) > 1 else ''
                targets.append((crop, part, cur))
    if not targets:
        print('No targets found in runs_delta_report.csv')
        return

    out_rows = []

    for crop, part, run in targets:
        print('Processing', run)
        preds = find_predictions(crop, part, run)
        if not preds:
            print('  predictions.csv not found for', run)
            continue
        pr_file = find_pr(crop, part, run)
        slices = []
        if pr_file:
            try:
                with open(pr_file, 'r', encoding='utf-8') as f:
                    pr = json.load(f)
                slices = list(pr.get('context', {}).get('ood_type_breakdown', {}).keys() or [])
            except Exception:
                slices = []

        # read predictions
        with open(preds, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames

        if not header:
            print('  empty predictions header for', run)
            continue

        # find available score columns per method
        method_cols = {}
        for m in METHODS:
            col = detect_score_columns(header, m)
            if col:
                method_cols[m] = col

        # also detect primary score column
        primary_col = None
        if 'ood_primary_score' in header:
            primary_col = 'ood_primary_score'

        # for each slice, compute metrics using scores if available
        for slice_name in slices:
            # select positive rows for this slice
            pos = [r for r in rows if (r.get('ood_type') == slice_name or r.get('sample_origin') == slice_name)]
            neg = [r for r in rows if (r.get('sample_origin') == 'in_distribution' or (r.get('ood_type') == '' and r.get('sample_origin') == 'in_distribution'))]
            if not pos or not neg:
                # try alternative: use rows where ood_type contains slice_name
                pos = [r for r in rows if slice_name in (r.get('ood_type') or '')]
            if not pos or not neg:
                # skip if not enough data
                continue
            for m, col in method_cols.items():
                # build y and scores: only include rows present in pos+neg
                selected = pos + neg
                y = [1]*len(pos) + [0]*len(neg)
                scores = []
                skip = False
                for r in selected:
                    v = r.get(col)
                    if v is None or v == '':
                        skip = True
                        break
                    try:
                        scores.append(float(v))
                    except Exception:
                        skip = True
                        break
                if skip or len(set(y)) < 2:
                    continue
                auroc = compute_auroc(y, scores)
                fpr95 = compute_fpr_at_tpr95(y, scores)
                out_rows.append({'run': run, 'crop': crop, 'part': part, 'slice': slice_name, 'method': m, 'auroc': auroc, 'fpr_at_tpr95': fpr95, 'pos_count': len(pos), 'neg_count': len(neg), 'score_column': col})

            # fallback: compute for primary score if method-specific columns missing
            if not method_cols and primary_col:
                selected = pos + neg
                y = [1]*len(pos) + [0]*len(neg)
                scores = []
                skip = False
                for r in selected:
                    v = r.get(primary_col)
                    if v is None or v == '':
                        skip = True
                        break
                    try:
                        scores.append(float(v))
                    except Exception:
                        skip = True
                        break
                if not skip and len(set(y)) > 1:
                    auroc = compute_auroc(y, scores)
                    fpr95 = compute_fpr_at_tpr95(y, scores)
                    out_rows.append({'run': run, 'crop': crop, 'part': part, 'slice': slice_name, 'method': 'primary', 'auroc': auroc, 'fpr_at_tpr95': fpr95, 'pos_count': len(pos), 'neg_count': len(neg), 'score_column': primary_col})

    # write output CSV
    if out_rows:
        with open(OUT, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['run','crop','part','slice','method','auroc','fpr_at_tpr95','pos_count','neg_count','score_column']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print('Wrote', OUT)
    else:
        print('No per-slice per-sample metrics computed (no suitable score columns found)')


if __name__ == '__main__':
    main()
