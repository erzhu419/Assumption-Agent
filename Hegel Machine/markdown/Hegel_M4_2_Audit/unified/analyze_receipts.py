"""Aggregate all 792 committed receipts, without rerunning the study."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

OUT = Path('/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified')
EVID = Path('/workspace/scratch/3f14ad2da0ed/repo_m42_fb4b23aaf/Hegel Machine/docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120')
rows = [json.loads(line) for line in (EVID / 'part4_unified_execution_receipts.jsonl').read_text().splitlines()]
report = json.loads((EVID / 'part4_unified_cross_pillar_lifecycle_and_breakeven.json').read_text())
groups = defaultdict(list)
for r in rows:
    if not r['is_out_of_scope']:
        groups[(r['arm'], r['mechanism_family'])].append(r)
summary = []
for (arm, family), rs in groups.items():
    summary.append({'arm': arm, 'family': family, 'runs': len(rs),
                    'strict_passes': sum(r['strict_pass'] for r in rs),
                    'false_confirmations': sum(r['false_confirmation'] for r in rs),
                    'recall_confirmations': sum(r['confirmed_via_recall'] for r in rs),
                    'walk_wins': sum(r['non_direct_walk_win'] for r in rs),
                    'mean_measurement_cost': statistics.mean(r['cost']['measurement_cost'] for r in rs),
                    'mean_composite_cost': statistics.mean(r['cost']['composite_cost'] for r in rs),
                    'mean_candidate_count': statistics.mean(r['evaluated_candidate_count'] for r in rs)})
reused = [r for r in rows if all(k in r['cost']['phase_breakdown'] for k in ('ordinary_confirm_0', 'ordinary_confirm_1', 'fallback_confirm'))]
pair = {}
for r in rows:
    if r['arm'] in ('Hegel_Unified_Continual_Forest', 'Hegel_NoMaintenance_SameHistory') and not r['is_out_of_scope']:
        pair.setdefault((r['seed'], r['target_id']), {})[r['arm']] = r
deltas = []
for (seed, target), ar in pair.items():
    h, n = ar['Hegel_Unified_Continual_Forest'], ar['Hegel_NoMaintenance_SameHistory']
    deltas.append({'seed': seed, 'target_id': target, 'family': h['mechanism_family'],
                   'composite_delta_no_maint_minus_hegel': n['cost']['composite_cost'] - h['cost']['composite_cost'],
                   'measurement_delta_no_maint_minus_hegel': n['cost']['measurement_cost'] - h['cost']['measurement_cost'],
                   'candidate_count_delta_no_maint_minus_hegel': n['evaluated_candidate_count'] - h['evaluated_candidate_count'],
                   'hegel_recalled': h['confirmed_via_recall'], 'no_maint_recalled': n['confirmed_via_recall']})
result = {
    'row_count': len(rows),
    'unique_seed_arm_target_keys': len({(r['seed'], r['arm'], r['target_id']) for r in rows}),
    'arms': sorted({r['arm'] for r in rows}),
    'seed_count': len({r['seed'] for r in rows}),
    'family_summary': summary,
    'dev_freeze': report['stage2']['dev_static_order_freeze'],
    'reused_confirmation_count': len(reused),
    'reused_confirmation_strict_passes': sum(r['strict_pass'] for r in reused),
    'reused_confirmation_false_predictions': sum(r['false_confirmation'] for r in reused),
    'reused_confirmation_rows': reused,
    'all_false_confirmations': [r for r in rows if r['false_confirmation']],
    'maintenance_deltas_per_row': deltas,
    'maintenance_delta_total': sum(d['composite_delta_no_maint_minus_hegel'] for d in deltas),
    'non_pi_maintenance_differences': [d for d in deltas if d['family'] != 'PiS_Adiabatic_CrossDomain_2D' and any(d[k] != 0 for k in ('composite_delta_no_maint_minus_hegel','measurement_delta_no_maint_minus_hegel','candidate_count_delta_no_maint_minus_hegel'))],
}
(OUT / 'receipts_analysis.json').write_text(json.dumps(result, indent=2))
print({k: result[k] for k in ('row_count','unique_seed_arm_target_keys','seed_count','reused_confirmation_count','reused_confirmation_strict_passes','reused_confirmation_false_predictions','maintenance_delta_total','non_pi_maintenance_differences')})
