"""Generate visualizations for run deltas and per-slice OOD metrics.

Usage:
  python scripts/visualize_runs.py [--all | --top N]

Outputs PNGs into .runtime_tmp/plots/
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RUNTIME_TMP = Path('.runtime_tmp')
PLOTS_DIR = RUNTIME_TMP / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def find_production_readiness(run_base_dir: Path):
    # Search for production_readiness.json under run_base_dir
    for p in run_base_dir.rglob('production_readiness.json'):
        return p
    return None


def plot_delta_fpr(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10,6))
    df_sorted = df.sort_values('delta_fpr', ascending=False)
    ax.bar(df_sorted['crop_part'], df_sorted['delta_fpr'], color='C1')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_ylabel('Delta OOD FPR (cur - prev)')
    ax.set_title('Delta OOD FPR by crop_part')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p = PLOTS_DIR / 'delta_fpr_bar.png'
    fig.savefig(p)
    plt.close(fig)
    print('Saved', p)


def plot_auroc_vs_fpr(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,6))
    sizes = (df['ood_samples_cur'].fillna(10).astype(float) / df['ood_samples_cur'].max()) * 200
    sc = ax.scatter(df['ood_auroc_cur'], df['ood_fpr_cur'], s=sizes, c='C0', alpha=0.8)
    for i, row in df.iterrows():
        ax.annotate(row['crop_part'], (row['ood_auroc_cur'], row['ood_fpr_cur']), textcoords='offset points', xytext=(5,5))
    ax.set_xlabel('OOD AUROC (current)')
    ax.set_ylabel('OOD FPR (current)')
    ax.set_title('OOD AUROC vs FPR (current runs)')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    p = PLOTS_DIR / 'auroc_vs_fpr_scatter.png'
    fig.savefig(p)
    plt.close(fig)
    print('Saved', p)


def plot_run_slices(production_path: Path, run_name: str):
    with open(production_path, 'r', encoding='utf-8') as f:
        pr = json.load(f)
    ood_type_breakdown = pr.get('ood_type_breakdown', {})
    # fallback: some runs write a dedicated ood_type_breakdown.json under artifacts/test or validation
    if not ood_type_breakdown:
        # try to discover an ood_type_breakdown.json nearby
        run_root = production_path.parents[3] if len(production_path.parents) > 3 else production_path.parent
        for p in run_root.rglob('ood_type_breakdown.json'):
            try:
                with open(p, 'r', encoding='utf-8') as ff:
                    ood_type_breakdown = json.load(ff)
                print('Loaded ood_type_breakdown from', p)
                break
            except Exception:
                continue
    # build dataframe
    rows = []
    for name, info in ood_type_breakdown.items():
        metrics = info.get('metrics', {})
        rows.append({
            'slice': name,
            'ood_auroc': metrics.get('ood_auroc'),
            'ood_fpr': metrics.get('ood_false_positive_rate'),
            'ood_samples': metrics.get('ood_samples')
        })
    if not rows:
        print('No ood_type_breakdown for', run_name)
        return
    df = pd.DataFrame(rows).sort_values('ood_fpr', ascending=False)

    # AUROC bar
    fig, ax = plt.subplots(figsize=(10, max(3, len(df)*0.5)))
    ax.barh(df['slice'], df['ood_auroc'], color='C2')
    ax.set_xlabel('OOD AUROC')
    ax.set_xlim(0,1)
    ax.set_title(f'{run_name} - per-slice OOD AUROC')
    plt.tight_layout()
    p1 = PLOTS_DIR / f'{run_name}_slice_auroc.png'
    fig.savefig(p1)
    plt.close(fig)
    print('Saved', p1)

    # FPR bar
    fig, ax = plt.subplots(figsize=(10, max(3, len(df)*0.5)))
    ax.barh(df['slice'], df['ood_fpr'], color='C3')
    ax.set_xlabel('OOD False Positive Rate')
    ax.set_xlim(0,1)
    ax.set_title(f'{run_name} - per-slice OOD FPR')
    plt.tight_layout()
    p2 = PLOTS_DIR / f'{run_name}_slice_fpr.png'
    fig.savefig(p2)
    plt.close(fig)
    print('Saved', p2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Plot slices for all runs')
    parser.add_argument('--top', type=int, default=4, help='Top N problematic runs by delta_fpr to plot')
    args = parser.parse_args()

    csvp = Path('.runtime_tmp') / 'runs_delta_report.csv'
    if not csvp.exists():
        print('CSV report not found at', csvp)
        return
    df = pd.read_csv(csvp)

    plot_delta_fpr(df)
    plot_auroc_vs_fpr(df)

    # select runs to plot
    if args.all:
        to_plot = df['current_run'].tolist()
    else:
        df_sorted = df.sort_values('delta_fpr', ascending=False)
        to_plot = df_sorted.head(args.top)['current_run'].tolist()

    for i, row in df[df['current_run'].isin(to_plot)].iterrows():
        crop_part = row['crop_part']
        cur = row['current_run']
        # crop_part is like 'apricot_leaf' -> split to crop/part
        if '_' in crop_part:
            parts = crop_part.split('_')
            crop = parts[0]
            part = '_'.join(parts[1:])
        else:
            # fallback
            crop = crop_part
            part = ''
        run_base = Path('runs') / crop / part / cur
        if not run_base.exists():
            # try without part folder (some repos use crop_part grouping)
            run_base = Path('runs') / crop_part / cur
        prod = find_production_readiness(run_base)
        if prod:
            try:
                plot_run_slices(prod, cur)
            except Exception as e:
                print('Failed plotting slices for', cur, 'err=', e)
        else:
            print('production_readiness.json not found for', cur, 'under', run_base)

if __name__ == '__main__':
    main()
