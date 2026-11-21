"""Read-only recomputation of committed M5 evidence; no benchmark execution."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',default=os.environ.get('HEGEL_AUDIT_REPO_ROOT','.'),help='Assumption-Agent checkout root, or its Hegel Machine subdirectory')
parser.add_argument('--evidence-dir',default=os.environ.get('HEGEL_AUDIT_EVIDENCE_DIR'),help='Committed evidence directory; defaults to repository docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120')
parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parent),help='Audit output directory')
args=parser.parse_args()
PROJECT=Path(args.repo_root).expanduser().resolve()
if (PROJECT/'Hegel Machine').is_dir():
    PROJECT=PROJECT/'Hegel Machine'
ROOT=Path(args.output_dir).expanduser().resolve()
ROOT.mkdir(parents=True,exist_ok=True)
EV=Path(args.evidence_dir).expanduser().resolve() if args.evidence_dir else PROJECT/'docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120'
rows = [json.loads(x) for x in (EV / 'part2_track1_anonymous_receipts.jsonl').read_text().splitlines()]
summary = json.loads((EV / 'part2_track1_anonymous_summary.json').read_text())
checks = []
for line in (EV / 'SHA256SUMS.txt').read_text().splitlines():
    expected, name = line.split('  ', 1)
    actual = hashlib.sha256((EV / name).read_bytes()).hexdigest()
    checks.append(dict(path=name, expected=expected, actual=actual, match=actual == expected))

def agg(rs):
    m = round(sum(r['measurement_cost'] for r in rs), 6)
    k = round(sum(r['compute_cost'] for r in rs), 6)
    ir = [r for r in rs if not r['is_out_of_scope']]
    oo = [r for r in rs if r['is_out_of_scope']]
    return dict(n=len(rs), in_scope=len(ir), in_scope_pass=sum(r['strict_pass'] for r in ir),
                oos=len(oo), oos_unknown=sum(r['status']=='scoped_unknown' for r in oo),
                false_confirmation=sum(r['false_confirmation'] for r in rs),
                candidates=sum(r['evaluated_candidate_count'] for r in rs),
                nfev=sum(r['total_nfev'] for r in rs), M=m, K=k, C=round(m+.05*k,6),
                sum_row_C=round(sum(r['composite_cost'] for r in rs),6))

arms = sorted({r['arm'] for r in rows})
families = sorted({r['subfamily'] for r in rows})
arm_aggs = {a: agg([r for r in rows if r['arm']==a]) for a in arms}
family_aggs = {f: {a:agg([r for r in rows if r['arm']==a and r['subfamily']==f]) for a in arms} for f in families}
seed_aggs = {str(s):{a:agg([r for r in rows if r['arm']==a and r['seed']==s]) for a in arms} for s in summary['seeds']}
compare_keys = {'total_tasks':'n','in_scope_count':'in_scope','in_scope_strict_pass':'in_scope_pass','oos_count':'oos','oos_honest_unknown':'oos_unknown','false_confirmations':'false_confirmation','total_evaluated_candidates':'candidates','total_scipy_nfev':'nfev','target_measurement_cost_total':'M','target_compute_cost_total':'K','target_composite_cost_total':'C'}
summary_mismatches = []
for a, ar in arm_aggs.items():
    for sk, ak in compare_keys.items():
        if abs(summary['arm_summaries'][a][sk]-ar[ak]) > 1e-5:
            summary_mismatches.append(dict(arm=a,key=sk,recorded=summary['arm_summaries'][a][sk],recomputed=ar[ak]))

full='Hegel_M5_Full_TemporalSubtree_Forest'
group2='Hegel_M5_Forest_Subtree_PPR'
baselines=['Strong_Dynamic_StateSpace_AllHistory','EqualAuth_SingleNode_Wormhole_Only','EqualAuth_Dynamic_Flat_BM25','EqualAuth_RawVariance_Probe_Ablation']
cost_differences={}
for base in baselines:
    treatment=full if base.startswith('Strong') else group2
    by_f={f:{k:round(family_aggs[f][base][k]-family_aggs[f][treatment][k],6) for k in ['M','K','C','nfev','candidates']} for f in families}
    difference={k:round(arm_aggs[base][k]-arm_aggs[treatment][k],6) for k in ['M','K','C','nfev','candidates']}
    paired=[]
    indexes={a:{(r['seed'],r['task_id']):r for r in rows if r['arm']==a} for a in (base,treatment)}
    for key,t in indexes[treatment].items():
        b=indexes[base][key]
        paired.append(b['composite_cost']-t['composite_cost'])
    cost_differences[base]=dict(treatment=treatment,total= difference,by_family=by_f,
        paired_treatment_cheaper=sum(d>1e-9 for d in paired),paired_tied=sum(abs(d)<=1e-9 for d in paired),paired_treatment_dearer=sum(d < -1e-9 for d in paired),
        max_source_formation_cost_for_observed_batch_break_even=difference['C'])

j=summary['flagship_joint_stream_J']
j_aggs={a:agg(rs) for a,rs in j['receipts_by_arm'].items()}
cold=j['cold_subprocess_continuation_J17_J20']
snapshot=json.loads((EV/'track1_anonymous_benchmark/joint_stream_J/relocatable_m5_joint_snapshot/post_j16_m5_forest_snapshot.json').read_text())
coarse_families={'B_adiabatic_coarse_valid','B_memory_kernel_fold'}
strong_rows=[r for r in rows if r['arm']=='Strong_Dynamic_StateSpace_AllHistory']
strong_repriced_M=sum(12*(1.0 if r['subfamily'] in coarse_families else 1.5) for r in strong_rows)
strong_C=arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['C']
full_C=arm_aggs[full]['C']
same_unit_price=dict(
    note='Accounting sensitivity only: same committed solver outputs, same fixed 12 Strong rows; reprice each Strong family at Hegel initial-row prices. Not an adaptive rerun and does not settle physical-channel comparability.',
    strong_original_M=arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['M'],
    strong_repriced_M=strong_repriced_M,
    price_component_removed=arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['M']-strong_repriced_M,
    remaining_measurement_gap=strong_repriced_M-arm_aggs[full]['M'],
    original_C_gap=round(strong_C-full_C,6),
    repriced_strong_C=round(strong_repriced_M+.05*arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['K'],6),
    remaining_C_gap=round(strong_repriced_M+.05*arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['K']-full_C,6),
    proportion_original_gap_measurement=(arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['M']-arm_aggs[full]['M'])/(strong_C-full_C),
    proportion_original_gap_same_row_price=(arm_aggs['Strong_Dynamic_StateSpace_AllHistory']['M']-strong_repriced_M)/(strong_C-full_C))
out=dict(commit='63b940a03205d0f5b36bf2403d274d74709a17fe',
    sha_checks=checks,sha_all_match=all(c['match'] for c in checks),
    receipt_count=len(rows),unique_arm_seed_task_count=len({(r['arm'],r['seed'],r['task_id']) for r in rows}),
    arm_count=len(arms),per_arm=arm_aggs,per_family=family_aggs,per_seed=seed_aggs,
    summary_mismatches=summary_mismatches,cost_differences=cost_differences,
    fixed_policy_price_sensitivity=same_unit_price,
    selected_classes_by_family_Hegel={f:dict(collections.Counter(r.get('selected_model_class') for r in rows if r['arm']==full and r['subfamily']==f)) for f in families},
    j_aggregates=j_aggs,j_cold_aggregate=agg(cold['cold_receipts']),j_cold_digest=cold['digest_check'],
    snapshot_structure_counts={k:len(v) for k,v in snapshot.items() if isinstance(v,(dict,list))},
    observed_formation_costs={a:round(summary['arm_summaries'][a]['lifecycle_composite_cost_with_formation']-summary['arm_summaries'][a]['target_composite_cost_total'],6) for a in arms})
(ROOT/'evidence_recomputed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ['sha_all_match','receipt_count','unique_arm_seed_task_count','arm_count','summary_mismatches','per_arm','observed_formation_costs','snapshot_structure_counts']},indent=2))
