#!/usr/bin/env python3
"""Focused off-benchmark controls for the advertised earliest-boundary guarantee.

Loads the unmodified fixed-commit modules through namespace packages only.
Counterexample tasks have the same public interface and budgets as the study;
they add a short early failure pocket, without supplying hidden truth to solvers.
This is a diagnostic regression control, not an additional performance benchmark.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    src = args.repo.resolve() / 'src' / 'hegel_machine'
    for name, path in [('hegel_machine', src), ('hegel_machine.cognition', src / 'cognition')]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    from hegel_machine.cognition.cross_domain_arms import HegelV166Researcher, StrongOrdinaryResearcher
    from hegel_machine.cognition.cross_domain_clusters import PublicOrderedBoundaryTask, CostMeter, make_cluster_a_suite, make_cluster_b_suite
    h = HegelV166Researcher()
    h.form_and_revise_from_experience('audit_boundary', make_cluster_a_suite(3026101), make_cluster_b_suite(3026101))
    o = StrongOrdinaryResearcher()
    cases = {
        'opposite_endpoints_short_early_pocket': [1 if i in (3, 4) or i >= 80 else 0 for i in range(128)],
        'same_endpoints_short_early_pocket': [1 if i in (3, 4) or 40 <= i < 56 else 0 for i in range(128)],
    }
    output = {'commit': '675f52612a9d0f1fb208da8bdb0552de3c571758', 'scope': __doc__, 'cases': []}
    for case, seq in cases.items():
        true_first = next(i for i in range(1, len(seq)) if seq[i] != seq[i-1])
        for name, researcher in [('Hegel_V16_6', h), ('O_strong', o)]:
            task = PublicOrderedBoundaryTask(task_id=case, surface_domain='audit_release_history', length=len(seq), cost_per_query=3.0,
                task_view={'goal_kind': 'earliest_global_threshold', 'target_metric': 'exact_adjacent_boundary', 'tolerance': 0.0, 'input_domain': {'type':'ordered_discrete', 'length':len(seq)}, 'stochastic_oracle_hint':False},
                meter=CostMeter(obs_budget=72, meas_budget=216.0, compute_budget=500.0),
                private_oracle_fn=lambda i, values=seq: values[i])
            result = researcher.solve_target(case, task)
            output['cases'].append({'case':case, 'arm':name, 'true_earliest':true_first, 'submission':result,
                'false_confirmation':result.get('status')=='completed' and result.get('boundary_right_index') != true_first,
                'queried_indices':[q[0] for q in task.query_log], 'costs':task.meter.snapshot()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{'case':r['case'],'arm':r['arm'],'true':r['true_earliest'],'claimed':r['submission'].get('boundary_right_index'),'false_confirmation':r['false_confirmation']} for r in output['cases']],indent=2))


if __name__ == '__main__':
    main()
