
from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

ROOT = HEGEL_REPO
EVIDENCE = ROOT / 'docs/evidence/v16_8_m2_open_workspace_wormhole_20261119'
OUT = AUDIT_OUTPUT_DIR

def jsonl(name):
    return [json.loads(line) for line in (EVIDENCE / f'{name}.jsonl').read_text().splitlines()]

targets = jsonl('target_episodes')
sources = jsonl('source_episodes')
ledger = jsonl('cost_ledger')
published = json.loads((EVIDENCE / 'paired_results.json').read_text())
manifest = json.loads((EVIDENCE / 'experiment_manifest.json').read_text())
errors = []

def eq(a, b, name, tol=1e-6):
    if abs(a-b) > tol:
        errors.append({'check': name, 'actual': a, 'expected': b})

hashes = {rel: hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest for rel,digest in manifest['module_sha256'].items()}
ledger_map = {r['record_id']: r for r in ledger}
for row in targets+sources:
    for key in ['measurement_cost','compute_cost','composite_cost','observation_cost']:
        if key in row['costs']:
            eq(row['costs'][key],ledger_map[row['record_id']][key],f"ledger:{row['record_id']}:{key}")
    eq(row['costs']['composite_cost'],row['costs']['measurement_cost']+0.05*row['costs']['compute_cost'],f"formula:{row['record_id']}")

cluster_summaries = []
for cs in published['heldout_cluster_summaries']:
    cid = cs['cluster_id']
    out = {'cluster_id':cid,'arms':{}}
    for arm, expected in cs['arms'].items():
        rows = [r for r in targets if r['cluster_id']==cid and r['arm']==arm]
        source_arm = 'Hegel_M2' if arm in ['Flat_EqualContent_CodeBank','Hegel_M2_no_horizontal_wormhole'] else arm
        src = [r for r in sources if r.get('split')=='heldout_eval' and r['cluster_id']==cid and r['arm']==source_arm]
        actual = {
            'strict_pass_count':sum(bool(r['strict_pass']) for r in rows),
            'false_confirmations':sum(bool(r['false_confirmation']) for r in rows),
            'source_composite_cost':sum(r['costs']['composite_cost'] for r in src),
            'target_measurement_cost':sum(r['costs']['measurement_cost'] for r in rows),
            'target_compute_cost':sum(r['costs']['compute_cost'] for r in rows),
            'target_composite_cost':sum(r['costs']['composite_cost'] for r in rows),
        }
        actual['end_to_end_total_composite_cost'] = actual['source_composite_cost']+actual['target_composite_cost']
        actual['by_family_target_composite'] = {fam:sum(r['costs']['composite_cost'] for r in rows if r['cluster_family']==fam) for fam in ['Cluster_M2A','Cluster_M2B','Cluster_M2C']}
        for key,value in actual.items():
            if isinstance(value,dict):
                for k,v in value.items():eq(v,expected[key][k],f'{cid}:{arm}:{key}:{k}')
            else:eq(value,expected[key],f'{cid}:{arm}:{key}')
        out['arms'][arm]=actual
    cluster_summaries.append(out)

comparisons={}
for key,comp in published['key_paired_comparisons'].items():
    left,right=comp['comparison'].split('_minus_')
    diffs=[]
    for cs in cluster_summaries:
        lv,rv=cs['arms'][left],cs['arms'][right]
        for attr in comp['metric'].split('.'):
            lv,rv=lv[attr],rv[attr]
        diffs.append(lv-rv)
    eq(statistics.mean(diffs),comp['mean_diff'],f'comparison:{key}',tol=0.000051)
    for i,(act,exp) in enumerate(zip(diffs,comp['per_cluster_diffs'])):eq(act,exp,f'{key}:{i}',tol=0.000051)
    comparisons[key]={'per_cluster_diffs':diffs,'mean_diff':statistics.mean(diffs)}

by_type_flat={}
for name in dict.fromkeys(r['target_name'] for r in targets):
    diffs=[]
    for cs in cluster_summaries:
        cid=cs['cluster_id']
        rows={r['arm']:r for r in targets if r['cluster_id']==cid and r['target_name']==name}
        h,f=rows['Hegel_M2'],rows['Flat_EqualContent_CodeBank']
        diffs.append(h['costs']['composite_cost']-f['costs']['composite_cost'])
    by_type_flat[name]={'per_cluster_diffs':diffs,'mean_diff':statistics.mean(diffs)}

dev_total=sum(r['composite_cost'] for r in ledger if r['split']=='development')
eq(dev_total,published['dev_calibration_summary']['dev_total_composite_cost'],'dev_total')
h_o_e2e=sum(cs['arms']['Hegel_M2']['end_to_end_total_composite_cost']-cs['arms']['O_strong_CodeSession']['end_to_end_total_composite_cost'] for cs in cluster_summaries)
single_e2e=[cs['arms']['Hegel_M2']['end_to_end_total_composite_cost']-cs['arms']['Hegel_M2_single_source_only']['end_to_end_total_composite_cost'] for cs in cluster_summaries]
summary={
 'scope':'Receipt recomputation only; no runtime claims here.',
 'manifest_hash_matches':hashes,
 'counts':{'source_rows':len(sources),'target_rows':len(targets),'ledger_rows':len(ledger)},
 'arithmetic_errors':errors,
 'cluster_summaries':cluster_summaries,
 'comparisons':comparisons,
 'Hegel_minus_Flat_decomposition':by_type_flat,
 'unreported_lifecycle':{
  'Hegel_minus_single_source_e2e_per_cluster':single_e2e,
  'Hegel_minus_single_source_e2e_mean':statistics.mean(single_e2e),
  'dev_gate_cost':dev_total,
  'Hegel_minus_ordinary_4_cluster_e2e_given_gate':h_o_e2e,
  'Hegel_minus_ordinary_including_Hegel_dev_cost_only':h_o_e2e+dev_total,
  'break_even_number_of_10_target_batches_if_same_per_batch_saving_persists':dev_total/(-statistics.mean([cs['arms']['Hegel_M2']['end_to_end_total_composite_cost']-cs['arms']['O_strong_CodeSession']['end_to_end_total_composite_cost'] for cs in cluster_summaries])),
 },
 'statistical_note':{'independent_parameter_noise_seeds':4,'target_types':10,'one_sided_sign_test_all_4_improve_p':0.5**4,'two_sided_sign_test_all_4_improve_p':2*0.5**4},
}
(OUT/'receipt_recomputation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['cluster_summaries','comparisons','Hegel_minus_Flat_decomposition']},ensure_ascii=False,indent=2))
