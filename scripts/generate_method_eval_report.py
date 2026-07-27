"""Generate a single HTML report aggregating per-run recommendations, per-slice metrics,
plots and thumbnails.

Usage: python scripts/generate_method_eval_report.py
Outputs: .runtime_tmp/report/method_eval_report.html
"""
import csv
import json
from pathlib import Path

RUNTIME = Path('.runtime_tmp')
OUT_DIR = RUNTIME / 'report'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'method_eval_report.html'

def read_csv(path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def esc(s):
    return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def main():
    recs = read_csv(RUNTIME / 'method_selection_recommendations.csv')
    per_slice = read_csv(RUNTIME / 'method_selection_per_slice.csv')
    sample_rocs = read_csv(RUNTIME / 'per_slice_sample_roc.csv')

    # group per run
    runs = {}
    for r in recs:
        run = r['run']
        runs.setdefault(run, {})
        runs[run]['rec'] = r
    for r in per_slice:
        run = r['run']
        runs.setdefault(run, {})
        runs[run].setdefault('slices', []).append(r)
    for r in sample_rocs:
        run = r['run']
        runs.setdefault(run, {})
        runs[run].setdefault('sample_rocs', []).append(r)

    html_lines = []
    html_lines.append('<!doctype html>')
    html_lines.append('<html><head><meta charset="utf-8"><title>Method Evaluation Report</title>')
    html_lines.append('<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px} h2{margin-top:30px} .run{border:1px solid #ddd;padding:12px;margin-bottom:18px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #eee;padding:6px;text-align:left} img{max-width:320px;max-height:240px}</style>')
    html_lines.append('</head><body>')
    html_lines.append(f'<h1>Method Evaluation Report</h1>')
    html_lines.append(f'<p>Generated from runtime outputs in <code>.runtime_tmp/</code></p>')

    for run, info in sorted(runs.items()):
        rec = info.get('rec') or {}
        html_lines.append(f'<div class="run"><h2>{esc(run)}</h2>')
        html_lines.append('<p><b>Recommendation:</b> ' + esc(rec.get('recommended_method','(none)')) + '</p>')
        if rec.get('rationale'):
            html_lines.append('<p><b>Rationale:</b> ' + esc(rec.get('rationale')) + '</p>')

        # per-slice table
        slices = info.get('slices') or []
        if slices:
            html_lines.append('<h3>Per-slice method metrics (from production_readiness)</h3>')
            html_lines.append('<table><thead><tr><th>slice</th><th>method</th><th>ood_auroc</th><th>ood_fpr</th><th>ood_samples</th></tr></thead><tbody>')
            for s in slices:
                html_lines.append('<tr><td>' + esc(s.get('slice') or s.get('ood_type') or '') + '</td><td>' + esc(s.get('method') or '') + '</td><td>' + esc(s.get('ood_auroc') or '') + '</td><td>' + esc(s.get('ood_false_positive_rate') or s.get('fpr') or '') + '</td><td>' + esc(s.get('ood_samples') or s.get('pos_count') or '') + '</td></tr>')
            html_lines.append('</tbody></table>')

        # per-sample ROC table
        sample_rocs = info.get('sample_rocs') or []
        if sample_rocs:
            html_lines.append('<h3>Per-slice AUROC and FPR@TPR95 (from predictions.csv)</h3>')
            html_lines.append('<table><thead><tr><th>slice</th><th>method</th><th>auroc</th><th>fpr_at_tpr95</th><th>pos</th><th>neg</th></tr></thead><tbody>')
            for s in sample_rocs:
                html_lines.append('<tr><td>' + esc(s.get('slice') or '') + '</td><td>' + esc(s.get('method') or '') + '</td><td>' + esc(s.get('auroc') or '') + '</td><td>' + esc(s.get('fpr_at_tpr95') or '') + '</td><td>' + esc(s.get('pos_count') or '') + '</td><td>' + esc(s.get('neg_count') or '') + '</td></tr>')
            html_lines.append('</tbody></table>')

        # plots
        plot_dir = RUNTIME / 'plots' / run
        if plot_dir.exists():
            html_lines.append('<h3>ROC plots</h3>')
            for p in sorted(plot_dir.iterdir()):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif'):
                    rel = p.relative_to(OUT_DIR.parent)
                    html_lines.append(f'<div><img src="../{rel.as_posix()}" alt="{esc(p.name)}"><div>{esc(p.name)}</div></div>')

        # thumbnails
        samples_dir = RUNTIME / 'samples' / run
        if samples_dir.exists():
            html_lines.append('<h3>Sample thumbnails (worst slice)</h3>')
            for p in sorted(samples_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    rel = p.relative_to(OUT_DIR.parent)
                    html_lines.append(f'<img src="../{rel.as_posix()}" alt="{esc(p.name)}">')

        html_lines.append('</div>')

    html_lines.append('</body></html>')

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print('Wrote', OUT_FILE)


if __name__ == '__main__':
    main()
