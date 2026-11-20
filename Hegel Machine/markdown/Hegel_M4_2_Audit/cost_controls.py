"""Bounded controls for M4.2 cost attribution. Does not change audited sources."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import pickle

import hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle as m
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cost'
OUT.mkdir(exist_ok=True)
built = pickle.loads((ROOT / 'unified/shared_built.pkl').read_bytes())
targets, _ = m.build_m4_2_target_suite(seed=20261120)
history = built['ordinary_history_forward']
orig_acquire = OpenWorkspaceTaskLab.acquire


def run(name, target, forest, *, open_fallback=False):
    acquired = []
    retrievals = []
    idx = HippoRAGWormholeIndex(forest, damping_alpha=0.65,
                              wormhole_equivalence_pooling=True, scale_aware=True)
    orig_retrieve = idx.retrieve

    def record_acquire(self, sequence, *, workspace=None, phase='probe'):
        obs = orig_acquire(self, sequence, workspace=workspace, phase=phase)
        acquired.append({'phase': phase, 'unit_cost': self.cost_per_obs,
                         'rows': [{'features': dict(r.features), 'target': r.target}
                                  for r in (obs or [])]})
        return obs

    def record_retrieve(*args, **kwargs):
        r = orig_retrieve(*args, **kwargs)
        retrievals.append({'query_compute_cost': r.get('query_compute_cost'),
                           'candidates': [c['node_id'] for c in r['candidates']]})
        return r

    with patch.object(OpenWorkspaceTaskLab, 'acquire', record_acquire), \
            patch.object(idx, 'retrieve', record_retrieve):
        rec = m.evaluate_m4_2_target_under_arm(
            target, arm_name=name, seed=20261120, work_dir=OUT / name,
            forest=forest, index=idx, ordinary_history_specs=None,
            fallback_pool_specs=history, open_code_agent_fallback=open_fallback)
    normalized = [{'phase': x['phase'], 'rows': x['rows']} for x in acquired]
    result = {'receipt': rec, 'actual_acquisitions': acquired,
              'actual_acquired_row_count': sum(len(x['rows']) for x in acquired),
              'acquisition_digest_without_prices': hashlib.sha256(
                  json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
              'retrievals': retrievals,
              'live_coarse_edge_endpoints': [[e['fine_node_id'], e['coarse_node_id']]
                                            for e in forest.coarse_graining_edges
                                            if not e.get('stale_endpoint_revision')]}
    (OUT / (name + '.json')).write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


results = {}
for name, edge_filter in [
    ('pi_complete_forest', lambda e: True),
    ('pi_only_unrelated_hydro_edge', lambda e: e['fine_node_id'] == 'hydro_micro_pore_1d'),
    ('pi_no_coarse_edges', lambda e: False),
]:
    f = deepcopy(built['forest'])
    f.coarse_graining_edges = [e for e in f.coarse_graining_edges if edge_filter(e)]
    results[name] = run(name, targets[16], f)

frozen = PhilosophicalForest()
for nid in ('stage0_monod_2d', 'stage0_murray_1d', 'hydro_micro_pore_1d', 'ode_micro_2mode_node'):
    frozen.nodes[nid] = deepcopy(built['forest_before_stage2_maintenance'].nodes[nid])
for name, is_open in [('pool_fallback', False), ('open_rediscovery', True)]:
    results[name] = run(name, targets[0], deepcopy(frozen), open_fallback=is_open)

summary = {}
for name, r in results.items():
    summary[name] = {k: r[k] for k in ['actual_acquired_row_count', 'acquisition_digest_without_prices',
                                       'live_coarse_edge_endpoints', 'retrievals']}
    summary[name]['reported_cost'] = r['receipt']['cost']
    summary[name]['selected_node_id'] = r['receipt']['selected_node_id']
    summary[name]['strict_pass'] = r['receipt']['strict_pass']
    summary[name]['heldout_rmse'] = r['receipt']['heldout_rmse']
    summary[name]['evaluated_candidate_count'] = r['receipt']['evaluated_candidate_count']
summary['equal_work_checks'] = {
    'unrelated_edge_vs_no_edge_acquisition_identical':
        results['pi_only_unrelated_hydro_edge']['acquisition_digest_without_prices'] ==
        results['pi_no_coarse_edges']['acquisition_digest_without_prices'],
    'open_vs_pool_acquisition_identical':
        results['open_rediscovery']['acquisition_digest_without_prices'] ==
        results['pool_fallback']['acquisition_digest_without_prices'],
}
(OUT / 'cost_control_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
