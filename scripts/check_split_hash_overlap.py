import json
import csv
import sys
from collections import defaultdict

def check_overlap(manifest_path, out_csv=None, min_shared=1):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    hash_map = defaultdict(lambda: defaultdict(list))
    for row in manifest.get('rows', []):
        h = row.get('exact_hash')
        split = row.get('split')
        rel = row.get('runtime_relative_path') or row.get('relative_path')
        if h and split:
            hash_map[h][split].append(rel)

    duplicates = []
    for h, splits in hash_map.items():
        if len(splits) > 1:
            duplicates.append((h, {s: len(p) for s,p in splits.items()}, splits))

    if out_csv and duplicates:
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['exact_hash','split_counts','split','relative_path'])
            for h, split_counts, splits in duplicates:
                for s, paths in splits.items():
                    for p in paths:
                        writer.writerow([h, json.dumps(split_counts), s, p])

    return duplicates

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/check_split_hash_overlap.py <path/to/split_manifest.json> [out.csv]')
        sys.exit(2)
    manifest = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else '.runtime_tmp/duplicate_hashes_report.csv'
    dups = check_overlap(manifest, out_csv=out)
    print(f'Found {len(dups)} hashes present in more than one split. Report written to {out} (if any).')
