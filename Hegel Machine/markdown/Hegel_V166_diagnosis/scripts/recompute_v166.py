#!/usr/bin/env python3
"""Offline receipt arithmetic for fixed 675f5261 V16.6, without running models.

Recomputes costs from JSONL rows and quality from saved evaluator measurements.
It cannot recompute curve RMSE from submissions, which omit predict_fn/coefficients.
The separate offline replay verifies those curves by executing the fixed simulator.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
from statistics import mean, pstdev

COMMIT = '675f52612a9d0f1fb208da8bdb0552de3c571758'
H = 'Hegel_V16_6'
FIELDS = ('observation_cost', 'measurement_cost', 'compute_cost', 'composite_cost',
          'model_calls', 'reported_tokens')
SOURCE_ALIASES = {
    'Hegel_no_horizontal_wormhole': H,
    'Hegel_object_removed_rebuild_allowed': H,
    'O_strong_same_history_rebuild_opt': H,
}


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def rounded(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def totals(cost_rows):
    return {f: sum(r.get(f, 0) for r in cost_rows) for f in FIELDS}


def verify_saved_quality(row):
    """Rejudge using saved public evaluation measurements, not saved strict_pass."""
    sub, ev, name = row['submission'], row['evaluation'], row['target_name']
    complete = sub.get('status') == 'completed'
    if not complete:
        return False, False
    if 'true_earliest' in ev:
        ok = (sub.get('boundary_right_index') == ev['true_earliest']
              and sub.get('guarantee_kind') != 'some_adjacent_flip')
    elif 'valid_flips' in ev:
        ok = sub.get('boundary_right_index') in ev['valid_flips']
    elif 'true_fault' in ev:
        ok = sub.get('boundary_right_index') == ev['true_fault']
    elif 'achieved_output' in ev:
        ok = (ev['achieved_output'] >= .97 * ev['required_output']
              and ev['actual_resource_spend'] <= 1.15 * ev['optimal_resource_spend'])
    elif 'rmse' in ev:
        tol = ev.get('tolerance', .15 if name == 'T_C2_service_burst_tail_latency' else .025)
        ok = ev['rmse'] is not None and ev['rmse'] <= tol
    else:
        raise ValueError(f'Unknown saved evaluator schema: {name}')
    return bool(ok), not bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', required=True, type=Path)
    ap.add_argument('--repo-root', type=Path, help='Optional Hegel Machine directory for manifest hash checks')
    ap.add_argument('--output', type=Path, default=Path('recomputed_v166.json'))
    args = ap.parse_args()
    e = args.evidence
    target = rows(e/'target_episodes.jsonl')
    source = rows(e/'source_episodes.jsonl')
    ledger = rows(e/'cost_ledger.jsonl')
    published = json.loads((e/'paired_results.json').read_text())
    manifest = json.loads((e/'experiment_manifest.json').read_text())
    clusters = list(dict.fromkeys(r['cluster_id'] for r in target))
    arms = list(dict.fromkeys(r['arm'] for r in target))
    order = list(dict.fromkeys(r['target_name'] for r in target))
    source_ledger = [r for r in ledger if r['phase'] != 'target']
    target_ledger = [r for r in ledger if r['phase'] == 'target']
    target_lookup = {(r['cluster_id'], r['arm'], r['target_name']): r for r in target}
    failures = []
    integrity = {'target_keys_unique': len(target_lookup) == len(target),
                 'cost_formula_mismatches': [], 'phase_sum_mismatches': [],
                 'target_ledger_mismatches': [], 'source_export_order_mismatches': [],
                 'published_summary_mismatches': [], 'quality_mismatches': []}
    for i, r in enumerate(ledger):
        if abs(r['composite_cost'] - (r['measurement_cost']+.05*r['compute_cost'])) > 1e-6:
            integrity['cost_formula_mismatches'].append(i)
        for f, p in [('observation_cost', 'obs'), ('measurement_cost', 'meas'), ('compute_cost', 'compute')]:
            if abs(r[f]-sum(b[p] for b in r['phase_breakdown'].values())) > 1e-6:
                integrity['phase_sum_mismatches'].append([i, f])
    for r in target_ledger:
        saved = target_lookup[(r['cluster_id'], r['arm'], r['episode_name'])]['costs']
        if any(abs(saved[f]-r[f]) > 1e-6 for f in FIELDS):
            integrity['target_ledger_mismatches'].append([r['cluster_id'], r['arm'], r['episode_name']])
    if len(source) != len(source_ledger):
        integrity['source_export_order_mismatches'].append('length_mismatch')
    else:
        for i, (s, l) in enumerate(zip(source, source_ledger)):
            if (s['cluster_id'] != l['cluster_id'] or s['episode_name'] != l['episode_name']
                    or any(abs(s['costs'][f]-l[f]) > 1e-6 for f in FIELDS)):
                integrity['source_export_order_mismatches'].append(i)
    for r in target:
        q, fc = verify_saved_quality(r)
        if q != r['strict_pass'] or fc != r['false_confirmation']:
            integrity['quality_mismatches'].append([r['cluster_id'], r['arm'], r['target_name']])
        if not q or fc:
            failures.append([r['cluster_id'], r['arm'], r['target_name'], r['evaluation']])

    per_cluster = {}
    for cid in clusters:
        per_cluster[cid] = {}
        for arm in arms:
            tr = [r for r in target if r['cluster_id'] == cid and r['arm'] == arm]
            sr = [r for r in source_ledger if r['cluster_id'] == cid
                  and r['arm'] == SOURCE_ALIASES.get(arm, arm)]
            st, tt = totals(sr), totals([r['costs'] for r in tr])
            per_cluster[cid][arm] = {
                'source': st, 'target': tt,
                'end_to_end': {f: st[f]+tt[f] for f in FIELDS},
                'source_episode_count_charged': len(sr), 'target_count': len(tr),
                'strict_pass_count': sum(verify_saved_quality(r)[0] for r in tr),
                'false_confirmations': sum(verify_saved_quality(r)[1] for r in tr),
                'by_family': {fam: totals([r['costs'] for r in tr if r['cluster_family']==fam])
                              for fam in ('Cluster_A', 'Cluster_B', 'Cluster_C')},
            }
    for pc in published['heldout_cluster_summaries']:
        cid = pc['cluster_id']
        for arm, pub in pc['arms'].items():
            rec = per_cluster[cid][arm]
            checks = {
                'source_composite_cost': rec['source']['composite_cost'],
                'target_composite_cost': rec['target']['composite_cost'],
                'target_measurement_cost': rec['target']['measurement_cost'],
                'target_compute_cost': rec['target']['compute_cost'],
                'end_to_end_total_composite_cost': rec['end_to_end']['composite_cost'],
                'strict_pass_count': rec['strict_pass_count'],
                'total_targets': rec['target_count'], 'false_confirmations': rec['false_confirmations'],
            }
            for k, v in checks.items():
                if abs(pub[k]-v)>1e-6:
                    integrity['published_summary_mismatches'].append([cid, arm, k, pub[k], v])

    comparisons = {}
    for right in [a for a in arms if a != H]:
        entry = {}
        for phase in ('source', 'target', 'end_to_end'):
            d = {f: [per_cluster[c][H][phase][f]-per_cluster[c][right][phase][f] for c in clusters]
                 for f in FIELDS}
            entry[phase] = {'per_cluster': d,
                            'mean': {f: mean(v) for f,v in d.items()},
                            'std_population_composite': pstdev(d['composite_cost']),
                            'win_tie_loss': {
                                'H_lower': sum(v < -1e-6 for v in d['composite_cost']),
                                'ties': sum(abs(v)<=1e-6 for v in d['composite_cost']),
                                'H_higher': sum(v>1e-6 for v in d['composite_cost'])}}
        entry['ordered_prefix_full_cost_differences'] = {}
        for n in (1,3,6,12,14):
            prefix = order[:n]
            diffs = []
            for cid in clusters:
                src_diff = per_cluster[cid][H]['source']['composite_cost']-per_cluster[cid][right]['source']['composite_cost']
                target_diff = sum(target_lookup[(cid,H,t)]['costs']['composite_cost']-
                                  target_lookup[(cid,right,t)]['costs']['composite_cost'] for t in prefix)
                diffs.append(src_diff+target_diff)
            entry['ordered_prefix_full_cost_differences'][str(n)] = {'per_cluster':diffs,'mean':mean(diffs)}
        entry['compute_weight_sensitivity_end_to_end'] = {
            str(w): entry['end_to_end']['mean']['measurement_cost']+w*entry['end_to_end']['mean']['compute_cost']
            for w in (0,.01,.05,.1,1,3)}
        dc = entry['end_to_end']['mean']['compute_cost']
        dm = entry['end_to_end']['mean']['measurement_cost']
        entry['positive_weight_zero_crossing'] = -dm/dc if dc and -dm/dc > 0 else None
        comparisons[H+'_minus_'+right] = entry

    per_target = {}
    for t in order:
        per_target[t] = {}
        for a in arms:
            rr = [r for r in target if r['target_name']==t and r['arm']==a]
            per_target[t][a] = {f:mean(r['costs'][f] for r in rr) for f in FIELDS}
            per_target[t][a]['strict_pass_count'] = sum(verify_saved_quality(r)[0] for r in rr)
            per_target[t][a]['count'] = len(rr)

    mean_arms = {a:{p:{f:mean(per_cluster[c][a][p][f] for c in clusters) for f in FIELDS}
                   for p in ('source','target','end_to_end')} for a in arms}
    source_export_keys = Counter((r['cluster_id'],r['arm'],r['episode_name']) for r in source)
    source_quality_by_correct_ledger_arm = {}
    if not integrity['source_export_order_mismatches']:
        for a in sorted(set(r['arm'] for r in source_ledger)):
            paired_rows = [(s,l) for s,l in zip(source,source_ledger) if l['arm']==a]
            source_quality_by_correct_ledger_arm[a] = {
                'n':len(paired_rows),'strict_pass_recorded':sum(bool(s['strict_pass']) for s,l in paired_rows)}
    if args.repo_root:
        integrity['manifest_module_sha256_matches'] = {
            p:hashlib.sha256((args.repo_root/p).read_bytes()).hexdigest()==h
            for p,h in manifest['module_sha256'].items()}

    result = {
        'commit_reviewed':COMMIT,
        'method':'Receipt arithmetic and quality reclassification from saved evaluator measurements; no scientific rerun in this script.',
        'quality_limits':'Saved RMSE rounded to 6 decimals and 2D metrics to 4; submitted predictor functions/coefficients absent. Offline replay is separate.',
        'counts': {'source_rows':len(source),'target_rows':len(target),'ledger_rows':len(ledger),
                   'clusters':clusters,'targets_per_arm':dict(Counter(r['arm'] for r in target)),
                   'source_ledger_arms':dict(Counter(r['arm'] for r in source_ledger)),
                   'source_export_duplicate_identity_keys':sum(n>1 for n in source_export_keys.values()),
                   'source_quality_by_correct_ledger_arm':source_quality_by_correct_ledger_arm},
        'integrity': integrity, 'quality_failures':failures,
        'source_cost_aliases_used_by_published_runner':SOURCE_ALIASES,
        'mean_costs_per_cluster_by_arm':mean_arms,
        'per_cluster':per_cluster,'comparisons':comparisons,'per_target':per_target,
        'target_order_for_prefixes':order,
        'statistical_scope': {
            'unit':'4 source-cluster replicates of the same predefined 14 task types, not 56 independent learned mechanisms.',
            'four_of_four_direction_only_sign_test_one_sided':1/16,
            'four_of_four_direction_only_sign_test_two_sided':2/16,
            'sign_test_qualification':'Illustration under independent exchangeable sign assumptions; does not establish these assumptions or provide a generalization theorem.',
            'pstdev_is_not_confidence_interval':True,
        },
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (e/'source_episodes.jsonl', e/'target_episodes.jsonl',e/'cost_ledger.jsonl',e/'paired_results.json',e/'experiment_manifest.json')},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rounded(result),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'counts':result['counts'],
                      'integrity':integrity,'mean_costs':rounded(mean_arms),
                      'mean_differences':{k:{p:rounded(v[p]['mean']) for p in ('source','target','end_to_end')} for k,v in comparisons.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
