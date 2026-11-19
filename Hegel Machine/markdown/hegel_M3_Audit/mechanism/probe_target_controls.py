"""Minimal end-to-end post-hoc controls on two eval04 target episodes.

The committed eval04 forest is reloaded exactly. The original frozen Stage-1
gate's full records were not published, so these controls explicitly hold its
admission at True. Both original arms demonstrably entered their recall loop
on these two episodes; all other target operations use upstream solve_target.
Original PPR / Flat2 outputs and logical cost are checked against publication.
No source training, LLM calls, or repository modifications occur.
"""
from copy import deepcopy
import json
from pathlib import Path

from probe_retrieval_factorization import (
    OUT, EVID, load_forest, HippoRAGWormholeIndex,
    make_m3_large_scale_suite,
)
from hegel_machine.cognition.m3_scale_universe import M3ForestResearcher


class MatchedAdmissionGate:
    def decide_stage1(self, *args, **kwargs):
        return {
            'admit_transfer': True,
            'mode': 'posthoc_admission_held_equal_to_original_observed_admission',
        }


def main():
    selected = {
        'T13_guard_two_hop_strict_rollover',
        'T20_simple_monod_membrane_flux',
    }
    published = {
        (r['target_name'], r['arm']): r
        for r in map(json.loads, (EVID / 'target_episodes.jsonl').read_text().splitlines())
        if r['cluster_id'] == 'm3_eval_04' and r['target_name'] in selected
    }
    suite = make_m3_large_scale_suite(7027404)
    rows = []
    mismatches = []
    variants = [
        ('PPR_original', 'hipporag_ppr', 2, False),
        ('PPR_alpha0_same_pool', 'hipporag_ppr', 2, False),
        ('Flat_original_2_trials', 'flat_bm25', 2, False),
        ('Flat_3_trials', 'flat_bm25', 3, False),
        ('PPR_delete_all_program_steps', 'hipporag_ppr', 2, True),
    ]
    for category, target, (fac, ev_fn) in suite['targets']:
        if target not in selected:
            continue
        for label, mode, trials, delete_steps in variants:
            if delete_steps and not target.startswith('T13_'):
                continue
            researcher = M3ForestResearcher(
                retrieval_mode=mode,
                max_recalled_trials=trials,
                gate=MatchedAdmissionGate(),
            )
            researcher.forest = load_forest()
            if delete_steps:
                for node in researcher.forest.nodes.values():
                    node['program_steps'] = []
            researcher.build_or_refresh_index()
            if label == 'PPR_alpha0_same_pool':
                researcher.index = HippoRAGWormholeIndex(researcher.forest, damping_alpha=0.0)
            lab = fac()
            ws = OUT / 'target_control_workspaces' / target / label
            submission = researcher.solve_target(target, lab, ws)
            evaluation = ev_fn(ws, submission, lab.observations)
            costs = lab.meter.snapshot()
            record = {
                'target': target,
                'variant': label,
                'submission': submission,
                'evaluation': evaluation,
                'costs': costs,
                'retrieval_diagnostic': researcher.retrieval_logs[-1],
                'observation_features': [dict(r.features) for r in lab.observations],
            }
            rows.append(record)
            pub_arm = {
                'PPR_original': 'Hegel_M3_HippoPPR',
                'Flat_original_2_trials': 'Flat_BM25_EqualMetadata',
            }.get(label)
            if pub_arm:
                p = published[(target, pub_arm)]
                checks = {
                    'model_sha256': (submission.get('model_sha256'), p['submission'].get('model_sha256')),
                    'selected_schema': (submission.get('selected_schema'), p['submission'].get('selected_schema')),
                    'strict_pass': (evaluation['strict_pass'], p['strict_pass']),
                    'heldout_rmse': (evaluation['rmse'], p['evaluation']['rmse']),
                    'observation_cost': (costs['observation_cost'], p['costs']['observation_cost']),
                    'measurement_cost': (costs['measurement_cost'], p['costs']['measurement_cost']),
                    'compute_cost': (costs['compute_cost'], p['costs']['compute_cost']),
                    'composite_cost': (costs['composite_cost'], p['costs']['composite_cost']),
                    'confirmed_via_recall': (record['retrieval_diagnostic']['confirmed_via_recall'], p['retrieval_diagnostic']['confirmed_via_recall']),
                }
                for key, (now, then) in checks.items():
                    if now != then:
                        mismatches.append({'target': target, 'variant': label, 'field': key, 'now': now, 'published': then})
    summary = [
        {
            'target': r['target'], 'variant': r['variant'],
            'strict_pass': r['evaluation']['strict_pass'],
            'selected_schema': r['submission'].get('selected_schema'),
            'heldout_rmse': r['evaluation']['rmse'],
            'measurement_cost': r['costs']['measurement_cost'],
            'compute_cost': r['costs']['compute_cost'],
            'composite_cost': r['costs']['composite_cost'],
            'wrong_candidate_trials': r['retrieval_diagnostic']['wrong_candidate_trials'],
            'confirmed_via_recall': r['retrieval_diagnostic']['confirmed_via_recall'],
            'model_sha256': r['submission'].get('model_sha256'),
        }
        for r in rows
    ]
    out = {
        'fixed_commit': '612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8',
        'scope': 'Post-hoc targeted controls on eval04 committed forest, only T13 and T20; original observed Stage-1 admission held equal; no new benchmark or general speed claim.',
        'published_baseline_reconstruction_mismatches': mismatches,
        'summary': summary,
        'records': rows,
    }
    (OUT / 'target_controls_result.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
