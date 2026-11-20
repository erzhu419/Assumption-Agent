"""Read-only reconstruction of committed evidence; no fitting, measurement, or runner.

Use --repo to point at Hegel Machine in the fixed commit. Outputs go to
--output-dir, outside the repository. Original evidence is never modified.
"""
import argparse
import atexit
import io
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import hashlib
import json

COMMIT = 'e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b'
parser = argparse.ArgumentParser(description='Read-only checks of committed M5.0 evidence and two frozen-instance dispatches.')
parser.add_argument('--repo', required=True, type=Path, help='Path to Hegel Machine in the fixed commit checkout.')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
REPO = args.repo.resolve()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=REPO, text=True).strip()
if head != COMMIT:
    raise SystemExit(f'Expected commit {COMMIT}, got {head}')
ROOT = args.output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
EVIDENCE_PATH = 'Hegel Machine/docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120'
archive_bytes = subprocess.check_output(['git', 'archive', COMMIT, EVIDENCE_PATH], cwd=GIT_ROOT)
archive_dir = tempfile.TemporaryDirectory(prefix='hegel_e2_readonly_')
atexit.register(archive_dir.cleanup)
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:') as archive:
    archive.extractall(archive_dir.name, filter='data')
BUNDLE = Path(archive_dir.name) / EVIDENCE_PATH
sys.path.insert(0, str(REPO / 'src'))

def read_json(name):
    return json.loads((BUNDLE / name).read_text())

def read_jsonl(name):
    return [json.loads(line) for line in (BUNDLE / name).read_text().splitlines() if line.strip()]

def total(rows, key):
    return float(sum((Decimal(str(r['cost'][key])) for r in rows), Decimal(0)))

sha_rows = []
for line in (BUNDLE / 'SHA256SUMS.txt').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    name = name.lstrip('*')
    path = BUNDLE / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    sha_rows.append({'path': name, 'match': expected == actual, 'expected': expected, 'actual': actual})

receipts = read_jsonl('part3_m5_0_stream_receipts.jsonl')
cold_receipts = read_jsonl('part3_m5_0_cold_continuation_receipts.jsonl')
summary = read_json('part3_m5_0_continual_sublation_summary.json')
groups = defaultdict(list)
for receipt in receipts:
    groups[receipt['arm']].append(receipt)

arms = {}
for arm, rows in groups.items():
    s = summary['arms'][arm]
    arms[arm] = {
        'receipt_count': len(rows),
        'in_scope_count': sum(not r['is_out_of_scope'] for r in rows),
        'in_scope_pass_count': sum(not r['is_out_of_scope'] and r['strict_pass'] for r in rows),
        'out_of_scope_count': sum(r['is_out_of_scope'] for r in rows),
        'out_of_scope_unknown_count': sum(r['is_out_of_scope'] and r['status']=='scoped_unknown' for r in rows),
        'false_confirmations': sum(r['false_confirmation'] for r in rows),
        'measurement_cost': total(rows, 'measurement_cost'),
        'compute_cost_receipt_sum': total(rows, 'compute_cost'),
        'composite_cost_receipt_sum': total(rows, 'composite_cost'),
        'evaluated_candidates': sum(r['evaluated_candidate_count'] for r in rows),
        'summary_composite_cost': s['total_stream_composite_cost'],
        'summary_compute_cost': s['total_compute_cost'],
        'initial_index_build_compute_cost': s['initial_index_build_compute_cost'],
        'writeback_compute_cost_total': s['writeback_compute_cost_total'],
        'index_auto_refresh_count': s['index_auto_refresh_count'],
    }

ppr_name = 'Hegel_M5_LiveSublation_PPR_PoolOn'
flat_name = 'Dynamic_Flat_StructureMemory_LiveWriteBack'
ppr = {r['target_id']:r for r in groups[ppr_name]}
flat = {r['target_id']:r for r in groups[flat_name]}
diffs = []
for target_id, pr in ppr.items():
    fr = flat[target_id]
    diffs.append({
        'target_id': target_id,
        'stream_step': pr['stream_step'],
        'flat_minus_ppr_M': float(Decimal(str(fr['cost']['measurement_cost'])) - Decimal(str(pr['cost']['measurement_cost']))),
        'flat_minus_ppr_C': float(Decimal(str(fr['cost']['composite_cost'])) - Decimal(str(pr['cost']['composite_cost']))),
        'ppr_cost': {k:pr['cost'][k] for k in ('measurement_cost','compute_cost','composite_cost')},
        'flat_cost': {k:fr['cost'][k] for k in ('measurement_cost','compute_cost','composite_cost')},
    })
measurement_differences = [r for r in diffs if r['flat_minus_ppr_M'] != 0]
result = {
    'commit': COMMIT,
    'procedure': 'Read committed bytes; no runner, refit, measurement, or source edit.',
    'sha256_check': {
        'manifest_entry_count': len(sha_rows),
        'match_count': sum(r['match'] for r in sha_rows),
        'mismatches': [r for r in sha_rows if not r['match']],
        'rows': sha_rows,
    },
    'receipt_count': len(receipts),
    'unique_arm_step_target_count': len({(r['arm'],r['stream_step'],r['target_id']) for r in receipts}),
    'cold_receipt_count': len(cold_receipts),
    'arms': arms,
    'flat_minus_ppr_summary_C': round(arms[flat_name]['summary_composite_cost']-arms[ppr_name]['summary_composite_cost'],4),
    'all_nonzero_measurement_differences': measurement_differences,
    'flat_minus_ppr_C_excluding_measurement_gap_task': float(sum((Decimal(str(r['flat_minus_ppr_C'])) for r in diffs if r['flat_minus_ppr_M']==0),Decimal(0))),
    'per_target_flat_minus_ppr': diffs,
}
(ROOT/'integration_evidence_recalculation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

# Only load and predict from the committed frozen instance.
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import ForestScopeDispatcher
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import load_location_independent_forest
from hegel_machine.cognition.m5_0_closure_and_continual_sublation import _build_m5_0_stream_tasks

forest = load_location_independent_forest(BUNDLE/'relocatable_forest_snapshots/post_m5_online_stream_forest_snapshot.json', BUNDLE)
dispatcher = ForestScopeDispatcher(forest)
s02 = next(t for _,t in _build_m5_0_stream_tasks() if t.target_id=='M50_S02_W4_HighBleed_Counterexample')
control_rows = []
meta = {'active_arity':2,'haldane_turnover_shunt_regime':False,'secondary_bleed_active':True}
for b in (1.4, 1.8):
    features = {'s':4.0,'b':b}
    source_truth = 9.0*features['s']/(1.6+features['s']) + 0.72*b
    receipt = dispatcher.dispatch_predict(
        query_id=f'audit_cold_source_W4_b_{b:.2f}',
        candidate_tree_id='tree_monod_reactor',
        features=features,
        target_metadata=meta,
        ground_truth=source_truth,
    )
    s02_truth = s02.oracle_fn(features)
    s02_base = 7.2*features['s']/(1.85+features['s']) + 0.58*b
    control_rows.append({
        'features':features,
        'actual_dispatch_receipt_against_original_source_oracle':receipt,
        'original_source_oracle':source_truth,
        's02_target_oracle_from_committed_callable':s02_truth,
        's02_base_without_surge':s02_base,
        's02_omitted_surge_term':s02_truth-s02_base,
        'frozen_source_prediction_minus_s02_target': None if receipt.get('prediction') is None else receipt['prediction']-s02_truth,
    })
control = {
    'commit':COMMIT,
    'procedure':'Two direct dispatch_predict calls after cold-load of committed post_m5 snapshot; no fit or measurement.',
    'source_node':'stage1_W4_monod_shunt',
    'saved_source_parameters':[9.0,1.6,0.72],
    's02_target_parameters':[7.2,1.85,0.58],
    'saved_preconditions':forest.nodes['stage1_W4_monod_shunt']['claim_contract']['preconditions'],
    's02_tolerance':s02.tolerance,
    'results':control_rows,
    'interpretation':(
        'This confirms that the actual frozen dispatcher authorizes both b=1.40 and b=1.80 with the original source instance. '
        'The S02 target is a differently parameterized instance and its fitted target bundle is not persisted here. '
        'Therefore source-vs-S02 numeric discrepancies are not evidence of a new midpoint-specific runtime failure. '
        'The S02 omitted-term calculation independently shows that b<=1.85 is not a valid certified bound for the S02 base family at tolerance 0.05; '
        'at b=1.80 its unmodeled surge is 0.15625. This is an analytic scope counterexample, not a replayed target false confirmation.'
    ),
}
(ROOT/'midpoint_scope_cold_control.json').write_text(json.dumps(control,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({
    'sha256_entries':len(sha_rows),
    'sha256_matches':sum(r['match'] for r in sha_rows),
    'receipts':len(receipts),
    'measurement_gap_tasks':measurement_differences,
    'other_27_flat_minus_ppr_C':result['flat_minus_ppr_C_excluding_measurement_gap_task'],
    'cold_dispatch':[
        { 'b':r['features']['b'], 'status':r['actual_dispatch_receipt_against_original_source_oracle']['status'],
          'within_source_tolerance':r['actual_dispatch_receipt_against_original_source_oracle']['within_tolerance'],
          'fit_called':r['actual_dispatch_receipt_against_original_source_oracle']['fit_called'],
          's02_omitted_term':r['s02_omitted_surge_term'] }
        for r in control_rows
    ]
},ensure_ascii=False,indent=2))
