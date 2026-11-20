"""Recalculate maintenance and quality directly from M4.4 Part2 receipts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--evidence', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(s) for s in (args.evidence / 'part2_m4_4_execution_receipts.jsonl').read_text().splitlines()]
summary = json.loads((args.evidence / 'part2_m4_4_continual_benchmark_summary.json').read_text())
selected = {}
for arm in ('Hegel_Unified_Continual_Forest', 'Hegel_NoMaintenance_SameHistory', 'Frozen_After_Stage0_OpenRediscovery'):
    selected[arm] = {(r['seed'], r['target_id']): r for r in rows if r['arm'] == arm and not r['is_out_of_scope']}
h = selected['Hegel_Unified_Continual_Forest']
n = selected['Hegel_NoMaintenance_SameHistory']
by = defaultdict(lambda: {'count': 0, 'gross_C_saved': 0.0})
for key, row in h.items():
    group = row['mechanism_family']
    if group.startswith('PiS_'):
        group += '__adiabatic' if int(row['target_id'].split('_')[1][1:]) in (21, 22) else '__quench'
    by[group]['count'] += 1
    by[group]['gross_C_saved'] += n[key]['cost']['composite_cost'] - row['cost']['composite_cost']
for group in by.values():
    group['gross_C_saved'] = round(group['gross_C_saved'], 4)
gross = sum(v['gross_C_saved'] for v in by.values())
upfront_s2 = summary['stage2']['composite_cost']
false_rows = [{k: r[k] for k in ('arm', 'seed', 'target_id', 'false_confirmation', 'heldout_rmse', 'selected_node_id')} for r in rows if r['false_confirmation']]
up0 = summary['stage0']['composite_cost']
up1 = summary['stage1']['composite_cost']
curves = []
ch, cn, cf = up0 + up1 + upfront_s2, up0 + up1, up0
cross_n = cross_f = None
for i, key in enumerate(h, 1):
    ch += h[key]['cost']['composite_cost']
    cn += n[key]['cost']['composite_cost']
    cf += selected['Frozen_After_Stage0_OpenRediscovery'][key]['cost']['composite_cost']
    if cross_n is None and ch <= cn:
        cross_n = i
    if cross_f is None and ch <= cf:
        cross_f = i
    curves.append({'completed': i, 'hegel': round(ch, 4), 'no_maintenance': round(cn, 4), 'frozen_open': round(cf, 4)})
result = {
    'maintenance_by_family_and_regime': dict(by), 'gross_saved': round(gross, 4),
    'stage2_upfront': upfront_s2, 'net_saved': round(gross - upfront_s2, 4),
    'Pi_share_of_gross_savings': sum(v['gross_C_saved'] for k, v in by.items() if k.startswith('PiS_')) / gross,
    'false_confirmation_rows': false_rows,
    'breakeven_vs_no_maintenance': cross_n, 'breakeven_vs_frozen_open': cross_f,
    'final_costs': curves[-1], 'cumulative_costs': curves,
}
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'cumulative_costs'}, ensure_ascii=False, indent=2))
