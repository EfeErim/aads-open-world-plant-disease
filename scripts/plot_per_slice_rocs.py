"""Plot per-slice ROC curves for available OOD scoring methods.

Saves PNGs under .runtime_tmp/plots/<run>/<slice>.png

Usage: python scripts/plot_per_slice_rocs.py
"""
from pathlib import Path
import csv
import json
import math
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
except Exception as e:
    print('Missing plotting dependencies:', e)
    print('Install matplotlib and scikit-learn in the environment.')
    sys.exit(1)

RUNTIME = Path('.runtime_tmp')
PLOTS = RUNTIME / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)

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
    candidates = []
    for h in header:
        lh = h.lower()
        if method in lh and 'score' in lh:
            candidates.append(h)
    for h in header:
        lh = h.lower()
        if ('ood' in lh or 'score' in lh) and method in lh and h not in candidates:
            candidates.append(h)
    return candidates[0] if candidates else None


def fpr_at_tpr(fpr, tpr, target=0.95):
    for fp, tp in zip(fpr, tpr):
        if tp >= target:
            return fp
    return fpr[-1]


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
            if cp and '_' in cp:
                crop, part = cp.split('_', 1)
            else:
                parts = cur.split('_')
                crop = parts[0]
                part = parts[1] if len(parts) > 1 else ''
            targets.append((crop, part, cur))
    return targets


def plot_run(run_tuple):
    crop, part, run = run_tuple
    print('Plotting', run)
    preds = find_predictions(crop, part, run)
    if not preds:
        print('  predictions.csv not found for', run)
        return
    pr = find_pr(crop, part, run)
    slices = []
    if pr:
        try:
            with open(pr, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            slices = list(doc.get('context', {}).get('ood_type_breakdown', {}).keys() or [])
        except Exception:
            slices = []

    with open(preds, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    if not header:
        print('  empty predictions for', run)
        return

    method_cols = {}
    for m in METHODS:
        col = detect_score_columns(header, m)
        if col:
            method_cols[m] = col

    primary_col = 'ood_primary_score' if 'ood_primary_score' in header else None

    # if no method-specific columns, fallback to primary
    use_methods = list(method_cols.keys()) if method_cols else (['primary'] if primary_col else [])

    for slice_name in slices:
        pos = [r for r in rows if (r.get('ood_type') == slice_name or r.get('sample_origin') == slice_name)]
        neg = [r for r in rows if (r.get('sample_origin') == 'in_distribution' or (r.get('ood_type') == '' and r.get('sample_origin') == 'in_distribution'))]
        if not pos or not neg:
            pos = [r for r in rows if slice_name in (r.get('ood_type') or '')]
        if not pos or not neg:
            # skip slices without enough data
            continue

        plt.figure(figsize=(6,6))
        plotted = 0
        for m in use_methods:
            if m == 'primary':
                col = primary_col
                label = 'primary'
            else:
                col = method_cols.get(m)
                label = m
            if not col:
                continue
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
            fpr, tpr, thr = roc_curve(y, scores)
            auc = roc_auc_score(y, scores)
            fpr95 = fpr_at_tpr(fpr, tpr, 0.95)
            plt.plot(tpr, 1-fpr, label=f"{label} AUROC={auc:.3f} FPR@TPR95={fpr95:.3f}")
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.xlabel('TPR')
        plt.ylabel('1 - FPR (specificity-like)')
        plt.title(f"{run} — {slice_name}")
        plt.legend(loc='lower left', fontsize='small')
        out_dir = PLOTS / run
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{slice_name}.png"
        plt.savefig(out_file, bbox_inches='tight', dpi=150)
        plt.close()
        print('  saved', out_file)


def main():
    delta = Path('.runtime_tmp') / 'runs_delta_report.csv'
    targets = load_targets_from_delta(delta)
    if not targets:
        print('No targets found in runs_delta_report.csv')
        return
    for t in targets:
        plot_run(t)


if __name__ == '__main__':
    main()
