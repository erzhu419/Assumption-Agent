"""Bounded, offline M4.1 closure controls. Writes only beside this file.

Uses the finite supported ExprNode/step DSL. Does not test arbitrary malicious Python.
Run with PYTHONPATH=<99b1fdfeb Hegel Machine>/src.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import hegel_machine.cognition.m4_multiscale_renormalization as m
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m4_1_unified_discovery_and_renormalization import _load_node_verified_operator_spec
from hegel_machine.cognition.m4_compositional_discovery import (
    compile_operator_spec_to_python, load_verified_authored_method, make_atomic_operator_spec,
)
from hegel_machine.cognition.open_workspace_wormhole import canonicalize_python_model_ast
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use, ScaleMap

OUT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HEGEL_AUDIT_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m41_99b1fdfeb/Hegel Machine'))
WORK = OUT / 'current_semantics_workspaces'
WORK.mkdir(exist_ok=True)


def run_steps_control(target, forest, fresh_points):
    f = deepcopy(forest)
    for node in f.nodes.values():
        for step in node.get('program_steps', []):
            if step.get('step') == 'subprocess_isolated_confirm':
                step['fresh_points'] = fresh_points
    logs = []
    original = m.MultiScaleWorkspaceTaskLab.acquire_channel
    def record(self, points, *, channel, phase):
        rows = original(self, points, channel=channel, phase=phase)
        logs.append({'phase': phase, 'channel': channel, 'point_count': len(points), 'points': points})
        return rows
    with patch.object(m.MultiScaleWorkspaceTaskLab, 'acquire_channel', record):
        result = m.evaluate_m4_target_under_arm(
            target, arm_name='Hegel_M4_MultiScale_HippoWalk', forest=f,
            index=HippoRAGWormholeIndex(f, damping_alpha=.65, wormhole_equivalence_pooling=True, scale_aware=True),
            stage1_gate=None, work_dir=WORK / f'fresh_points_{fresh_points}',
        )
    result['cost'].pop('wall_clock_seconds', None)
    return {'declared_fresh_points': fresh_points, 'result': result, 'acquisitions': logs,
            'actual_confirm_count': sum(x['point_count'] for x in logs if x['phase'].startswith('confirm'))}


def auth(node, scope):
    return authorize_abstraction_use(
        requested_tier='bounded_trial', task_view={'goal_kind': 'open_workspace_mechanism_prediction', 'tolerance': .1},
        target_features={'audit_scope': scope}, claim_contract=node['claim_contract'], scale_map=node['scale_map'],
    )


def branch_verdict(node, query):
    # Scope non-membership is UNK, not empirical law refutation. This is the same
    # disjoint-domain OR use that disjunctive_branches is intended to preserve.
    branches = node['claim_contract'].get('disjunctive_branches') or [node['claim_contract']]
    return 'ACC' if any(all(query.get(p['field']) == p['value'] for p in b['preconditions']) for b in branches) else 'UNK'


def main():
    result = {'actual_commit': '99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c'}
    forest, _ = m.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    targets, _ = m.build_m4_cluster_targets(cluster_idx=0)
    p1 = run_steps_control(targets[0], forest, 1)
    p2 = run_steps_control(targets[0], forest, 2)
    result['valid_program_parameter_consumption'] = {
        'fresh_1': p1, 'fresh_2': p2,
        'equal_behavior_except_declared_value': p1['result'] == p2['result'] and p1['acquisitions'] == p2['acquisitions'],
    }

    f = deepcopy(forest)
    a_id, b_id = 'aq_macro_river_tree', 'dist_node_00'
    # Two identical supported laws, but distinct legitimate domains.
    for nid, scope in [(a_id, 'A'), (b_id, 'B')]:
        node = f.nodes[nid]
        node['observation_map'] = deepcopy(f.nodes[a_id]['observation_map'])
        node['scale_map'] = deepcopy(f.nodes[a_id]['scale_map'])
        node['claim_contract']['conclusion_or_action'] = deepcopy(f.nodes[a_id]['claim_contract']['conclusion_or_action'])
        node['claim_contract']['preconditions'] = [{'field': 'audit_scope', 'op': '==', 'value': scope}]
    before = {'A_original': auth(f.nodes[a_id], 'A'), 'B_original': auth(f.nodes[b_id], 'B')}
    rep = f.compress_compatible_nodes(
        keeper_node_id=a_id, redundant_node_id=b_id,
        evaluation_queries=[{'audit_scope': 'A'}, {'audit_scope': 'B'}], evaluator_fn=branch_verdict,
    )
    after = f.nodes[a_id]
    result['fidelity_real_authorizer_consumption'] = {
        'same_predictive_law': True, 'accepted_by_fidelity': rep['accepted'],
        'validated_and_committed_predicate_verdicts': {s: branch_verdict(after, {'audit_scope': s}) for s in ('A','B')},
        'original_authorization': before,
        'installed_authorization_A': auth(after, 'A'), 'installed_authorization_B': auth(after, 'B'),
        'installed_top_level_preconditions': after['claim_contract']['preconditions'],
        'disjunctive_branch_count': len(after['claim_contract']['disjunctive_branches']),
        'redundant_archived': f.nodes[b_id]['utility_ledger']['archived_low_utility'],
    }

    # Exactly the supported compiler and integrity fields used by the new test;
    # no confirmation, oracle, observation or source workspace is supplied.
    spec = make_atomic_operator_spec('michaelis_1d')
    code = compile_operator_spec_to_python(spec, role_names=['x'], lens={'sign':1.,'scale':1.,'offset':0.})
    ast_info = canonicalize_python_model_ast(code)
    payload = {
        'operator_spec': spec, 'role_names': ['x'], 'ast_skeleton_digest': ast_info['ast_skeleton_digest'],
        'skeleton_status': ast_info['skeleton_status'], 'model_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'authored_model_source': code, 'confirmation_digest': 'no_confirmation_was_run', 'confirmed_ok': True,
    }
    art = WORK / 'self_consistent_without_confirmation.json'
    art.write_text(json.dumps(payload, indent=2))
    result['authored_artifact_confirmation_provenance'] = {
        'loader_accepts': load_verified_authored_method(art) is not None,
        'confirmation_performed': False, 'observations_supplied': 0,
        'provided_confirmation_digest': payload['confirmation_digest'],
        'scope': 'Internally consistent metadata, not external confirmation provenance. Trusted present-day study generation does run confirmation.',
    }

    ep = REPO / 'docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/persisted_stage1_authored_workspaces'
    w1_path = ep / 'stage1_W1_haldane_shunt' / 'authored_method.json'
    w2_path = ep / 'stage1_W2_osc_shunt_4p' / 'authored_method.json'
    w1, w2 = json.loads(w1_path.read_text()), json.loads(w2_path.read_text())
    stable_path = WORK / 'artifact_swapped_in_place.json'
    stable_path.write_text(json.dumps(w1, indent=2))
    node = {'claim_contract': {'conclusion_or_action': {
        'operator_tag': 'op_composed_haldane_shunt_2d', 'operator_spec': deepcopy(w1['operator_spec']),
        'ast_skeleton_digest': w1['ast_skeleton_digest'], 'authored_artifact_path': str(stable_path),
    }}}
    before_spec = _load_node_verified_operator_spec(node)
    stable_path.write_text(json.dumps(w2, indent=2))
    after_spec = _load_node_verified_operator_spec(node)
    result['node_to_artifact_binding'] = {
        'before_returns_registered_W1': before_spec == w1['operator_spec'],
        'after_accepts_other_verified_W2': after_spec == w2['operator_spec'],
        'node_expected_ast_digest': node['claim_contract']['conclusion_or_action']['ast_skeleton_digest'],
        'replacement_artifact_ast_digest': w2['ast_skeleton_digest'],
        'digest_changed': w1['ast_skeleton_digest'] != w2['ast_skeleton_digest'],
        'registered_node_claim_unchanged': node['claim_contract']['conclusion_or_action']['operator_spec'] == w1['operator_spec'],
    }

    # Demonstrate documented API boundary: 'revalidate' is a rebind operation;
    # it does not itself discharge the closure proof obligation.
    vf = deepcopy(forest)
    before_cg = HippoRAGWormholeIndex(vf, damping_alpha=.65).edge_counts['coarse_graining_edges']
    new_sm = ScaleMap(coarse_graining_op='pi_audit_changed', window_or_partition={'scale_regime':'meso'},
        preserved_queries=['open_workspace_mechanism_prediction'], closure_scope='not_closed',
        proposed_discarded_variables=['fast_micro_mode'], verified_discarded_in_scope=[], error_budget=.25).to_dict()
    vf.empirical_revise_node('aq_meso_hex_cell', witness_episode_id='closure_loss', cause='abstraction_closure_loss', updated_scale_map=new_sm)
    after_stale = HippoRAGWormholeIndex(vf, damping_alpha=.65).edge_counts['coarse_graining_edges']
    reb = vf.revalidate_vertical_edges_after_revision('aq_meso_hex_cell', witness_episode_id='arbitrary_label', reason='not_a_measured_certificate')
    after_rebind = HippoRAGWormholeIndex(vf, damping_alpha=.65).edge_counts['coarse_graining_edges']
    result['vertical_revalidation_boundary'] = {
        'before': before_cg, 'after_closure_loss': after_stale, 'after_label_only_rebind': after_rebind,
        'scale_map_still_not_closed': vf.nodes['aq_meso_hex_cell']['scale_map']['closure_scope'] == 'not_closed',
        'rebind_return': reb,
        'scope': 'Does not establish an end-to-end wrong final grant; final ScaleMap gate remains separate.',
    }
    (OUT / 'current_semantics_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({
        'valid_programs_equal': result['valid_program_parameter_consumption']['equal_behavior_except_declared_value'],
        'actual_points_for_declared_2': p2['actual_confirm_count'],
        'fidelity': result['fidelity_real_authorizer_consumption'],
        'artifact_confirmation': result['authored_artifact_confirmation_provenance'],
        'artifact_binding': result['node_to_artifact_binding'],
        'vertical': result['vertical_revalidation_boundary'],
    }, indent=2))


if __name__ == '__main__':
    main()
