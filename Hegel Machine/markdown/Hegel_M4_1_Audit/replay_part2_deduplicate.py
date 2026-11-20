"""Read-only audit: capture study locals, preserve raw rows, and deduplicate exact target identities.

Runs the unchanged committed Part 2 function in an independent temporary workspace.
No source or committed evidence is modified.
"""
from __future__ import annotations
import importlib.util
import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path('/workspace/scratch/3f14ad2da0ed/repo_m41_99b1fdfeb/Hegel Machine'))
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'part2')
args = parser.parse_args()
REPO, OUT = args.repo.resolve(), args.out.resolve()
sys.path.insert(0, str(REPO / 'src'))
sp = importlib.util.spec_from_file_location('committed_study', REPO / 'scripts/run_v17_3_m4_1_unified_study.py')
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
OUT.mkdir(parents=True, exist_ok=True)

inventory = []
for cid, seed in m.M3_1_EVAL_SEEDS:
    suite = m.make_m3_1_factorized_walk_suite(seed, target_k=36)
    targets, walk = suite['targets'], suite['walk_targets']
    inventory.append({
        'cluster_id': cid, 'seed': seed, 'targets_count': len(targets),
        'walk_count': len(walk), 'unique_target_names': len({t[1] for t in targets + walk}),
        'walk_is_same_tuple_object_in_targets': [any(t is w for t in targets) for w in walk],
        'target_names': [t[1] for t in targets],
    })
(OUT / 'suite_identity_inventory.json').write_text(json.dumps(inventory, indent=2))

captured = {}
def profile(frame, event, arg):
    if event == 'return' and frame.f_code is m.run_part2_complete_m3_1_2x2_factorization.__code__:
        captured.update(frame.f_locals['agg'])

sys.setprofile(profile)
try:
    with TemporaryDirectory(prefix='m41_audit_part2_') as tmp:
        summary = m.run_part2_complete_m3_1_2x2_factorization(Path(tmp))
finally:
    sys.setprofile(None)

def summarize(rows):
    assoc = [r for r in rows if r['target'].startswith(('T01_', 'T02_', 'T03_', 'T04_', 'T05_', 'T06_'))]
    walk = [r for r in rows if r['target'].startswith(('T25_', 'T26_', 'T27_', 'T28_'))]
    return {
        'n': len(rows), 'strict_pass_count': sum(r['strict_pass'] for r in rows),
        'mean_measurement_cost': mean(r['measurement_cost'] for r in rows),
        'mean_compute_cost': mean(r['compute_cost'] for r in rows),
        'mean_composite_cost': mean(r['composite_cost'] for r in rows),
        'assoc_count': len(assoc), 'assoc_mean_composite_cost': mean(r['composite_cost'] for r in assoc),
        'walk_count': len(walk), 'walk_wins': sum(r['walk_win'] for r in walk),
        'walk_mean_composite_cost': mean(r['composite_cost'] for r in walk),
    }

results = {'committed_function_summary': summary, 'cells': {}}
for arm, rows in captured.items():
    unique = {}
    duplicates = []
    for row in rows:
        key = (row['cluster_id'], row['target'])
        if key in unique:
            duplicates.append({'key': list(key), 'exact_same_record': unique[key] == row,
                               'first': unique[key], 'second': row})
        else:
            unique[key] = row
    results['cells'][arm] = {
        'raw': summarize(rows), 'unique': summarize(list(unique.values())),
        'duplicate_count': len(duplicates),
        'all_duplicate_records_exactly_equal': all(d['exact_same_record'] for d in duplicates),
        'duplicates': duplicates,
    }
(OUT / 'all_496_records.json').write_text(json.dumps(captured, indent=2))
(OUT / 'deduplicated_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps({k: {a:b for a,b in v.items() if a != 'duplicates'} for k,v in results['cells'].items()}, indent=2))
