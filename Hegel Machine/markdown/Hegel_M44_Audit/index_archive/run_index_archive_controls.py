"""Read-only M4.4 audit; restore committed snapshots, no forest formation or model API.

All mutations are in independent in-memory copies; two bounded target evaluations
write authored working files under this audit directory. Never writes repo/evidence.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

CHECKOUT_ROOT = Path(os.environ.get(
    'HEGEL_AUDIT_REPO',
    '/workspace/scratch/3f14ad2da0ed/repo_m44_b965f022b',
)).expanduser().resolve()
# Accept either the git checkout root or its existing "Hegel Machine" subdirectory.
REPO = CHECKOUT_ROOT / 'Hegel Machine' if (CHECKOUT_ROOT / 'Hegel Machine').is_dir() else CHECKOUT_ROOT
OUT = Path(os.environ.get(
    'HEGEL_AUDIT_OUT', str(Path(__file__).resolve().parent),
)).expanduser().resolve()
sys.path.insert(0, str(REPO / 'src'))
sys.dont_write_bytecode = True

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import ForestScopeDispatcher
from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
    CompiledRelationalCueTable, build_m4_3_target_suite, evaluate_m4_3_target_under_arm,
)
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    load_location_independent_forest, verify_forest_snapshot_digests,
)
from hegel_machine.cognition.wormhole_forest import canonical_digest

EV = REPO / 'docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120'
SNAPS = EV / 'relocatable_forest_snapshots'
COARSE = 'ode_meso_schur_pi_thermal'
FINE = 'ode_micro_2mode_node'


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def load(name='post_stage2_forest_snapshot.json'):
    return load_location_independent_forest(SNAPS / name, EV)


def all_state_digest(f):
    return hashlib.sha256(json.dumps(f.__dict__, sort_keys=True).encode()).hexdigest()


def artifact_records(f):
    for nid, n in f.nodes.items():
        yield 'live:' + nid, n
    for nid, hist in f.node_history.items():
        for i, n in enumerate(hist):
            yield f'history:{nid}:{i}', n


def archive_checks():
    result = {}
    fields = ('task_view', 'observation_map', 'scale_map', 'claim_contract', 'program_steps')
    for sp in sorted(SNAPS.glob('*forest_snapshot.json')):
        f = load(sp.name)
        missing_paths, history_mismatches, branch_missing = [], [], []
        n_history = n_artifacts = 0
        for key, n in artifact_records(f):
            if key.startswith('history:'):
                n_history += 1
                actual = canonical_digest({k:n[k] for k in fields})
                if actual != n['semantic_digest']:
                    history_mismatches.append({'key':key, 'computed':actual, 'recorded':n['semantic_digest']})
            cc = n['claim_contract']
            acts = [('outer', cc.get('conclusion_or_action', {}))]
            acts += [(f'branch:{i}', b.get('conclusion_or_action', {})) for i,b in enumerate(cc.get('disjunctive_branches', []))]
            for part, act in acts:
                if not act.get('authored_artifact_path'):
                    continue
                n_artifacts += 1
                path = f.resolve_artifact_path(act['authored_artifact_path'])
                if not path.is_file():
                    missing_paths.append({'key':key, 'part':part, 'logical_path':act['authored_artifact_path']})
                if not act.get('confirmed_instance_digest'):
                    branch_missing.append({'key':key, 'part':part})
        cross_instance = []
        for e in f.coarse_graining_edges:
            a = f.nodes[e['coarse_node_id']]['claim_contract']['conclusion_or_action']
            vals = [a.get('confirmed_instance_digest'), e.get('confirmed_instance_digest'), e['empirical_closure_certificate'].get('confirmed_instance_digest')]
            if len(set(vals)) != 1:
                cross_instance.append({'coarse_node':e['coarse_node_id'], 'identities':vals})
        result[sp.name] = {
            'disk_check':verify_forest_snapshot_digests(json.loads(sp.read_text())),
            'loaded_check':verify_forest_snapshot_digests(f),
            'history_node_count':n_history, 'history_semantic_mismatches':history_mismatches,
            'live_and_history_artifact_actions_count':n_artifacts,
            'missing_resolved_artifact_paths':missing_paths,
            'missing_expected_instance_actions':branch_missing,
            'edge_certificate_instance_mismatches':cross_instance,
            'snapshot_keys':sorted(json.loads(sp.read_text())),
        }
    return result


def certificate_controls():
    s0 = load('true_stage0_forest_snapshot.json')
    s2 = load()
    old_cert = deepcopy(next(e['empirical_closure_certificate'] for e in s0.coarse_graining_edges if e['coarse_node_id'] == COARSE))
    new_cert = deepcopy(next(e['empirical_closure_certificate'] for e in s2.coarse_graining_edges if e['coarse_node_id'] == COARSE))
    out = {'genuine_instances':{'stage0':old_cert['confirmed_instance_digest'], 'stage2':new_cert['confirmed_instance_digest']}}

    rejected = deepcopy(s2)
    before = all_state_digest(rejected)
    r = rejected.revalidate_vertical_edges_with_certificate(COARSE, witness_episode_id='audit_real_s0_cert_on_s2', closure_certificate=old_cert)
    out['revalidate_cross_instance'] = {'result':r, 'entire_forest_state_unchanged':before == all_state_digest(rejected)}

    rejected_branch = deepcopy(s2)
    proposed = deepcopy(rejected_branch.nodes[COARSE]['claim_contract'])
    assert proposed.get('disjunctive_branches')
    proposed['disjunctive_branches'][0]['conclusion_or_action'] = deepcopy(s0.nodes[COARSE]['claim_contract']['conclusion_or_action'])
    before = all_state_digest(rejected_branch)
    r = rejected_branch.revalidate_vertical_edges_with_certificate(COARSE, witness_episode_id='audit_real_s0_branch_on_s2', closure_certificate=new_cert, updated_claim_contract=proposed)
    out['revalidate_cross_branch_instance'] = {'result':r, 'entire_forest_state_unchanged':before == all_state_digest(rejected_branch)}

    accepted = deepcopy(s2)
    proposed = deepcopy(accepted.nodes[COARSE]['claim_contract'])
    old_action = deepcopy(s0.nodes[COARSE]['claim_contract']['conclusion_or_action'])
    proposed['conclusion_or_action'] = deepcopy(old_action)
    for br in proposed.get('disjunctive_branches', []):
        br['conclusion_or_action'] = deepcopy(old_action)
    v_before = accepted.nodes[COARSE]['version']
    r = accepted.revalidate_vertical_edges_with_certificate(COARSE, witness_episode_id='audit_valid_atomic_rebind_to_existing_s0', closure_certificate=old_cert, updated_claim_contract=proposed, updated_scale_map=deepcopy(accepted.nodes[COARSE]['scale_map']))
    edge = next(e for e in accepted.coarse_graining_edges if e['coarse_node_id'] == COARSE)
    out['revalidate_valid_atomic_change'] = {'result':r, 'version_before':v_before, 'version_after':accepted.nodes[COARSE]['version'], 'node_instance':accepted.nodes[COARSE]['claim_contract']['conclusion_or_action']['confirmed_instance_digest'], 'edge_instance':edge['confirmed_instance_digest'], 'certificate_instance':edge['empirical_closure_certificate']['confirmed_instance_digest'], 'digest_check':verify_forest_snapshot_digests(accepted)}

    fresh_edge = deepcopy(s2)
    fresh_edge.coarse_graining_edges = [e for e in fresh_edge.coarse_graining_edges if e['coarse_node_id'] != COARSE]
    no_edge_control = deepcopy(fresh_edge)
    before = all_state_digest(fresh_edge)
    n_before = len(fresh_edge.coarse_graining_edges)
    bad_edge = fresh_edge.add_coarse_graining_edge(fine_node_id=FINE, coarse_node_id=COARSE, scale_map=deepcopy(fresh_edge.nodes[COARSE]['scale_map']), empirical_closure_certificate=old_cert)
    out['add_real_cross_instance_edge'] = {
        'raised':False, 'edge_count_before':n_before, 'edge_count_after':len(fresh_edge.coarse_graining_edges),
        'entire_forest_state_unchanged':before == all_state_digest(fresh_edge),
        'node_instance':fresh_edge.nodes[COARSE]['claim_contract']['conclusion_or_action']['confirmed_instance_digest'],
        'edge_instance':bad_edge['confirmed_instance_digest'],
        'certificate_instance':bad_edge['empirical_closure_certificate']['confirmed_instance_digest'],
        'edge_is_stale':bad_edge['stale_endpoint_revision'],
        'consumer_live_fine_children':[n['node_id'] for n in fresh_edge.fine_scale_children_of(COARSE)],
        'snapshot_digest_check_still_passes':verify_forest_snapshot_digests(fresh_edge),
    }
    targets, _ = build_m4_3_target_suite(seed=20301120)
    target = targets[20]
    assert target.is_multiscale_two_channel
    for label, f in [('no_coarse_edge_control',no_edge_control), ('cross_instance_edge_control',fresh_edge)]:
        rec = evaluate_m4_3_target_under_arm(target, arm_name='audit_' + label, seed=20301120, work_dir=OUT / 'work' / label, forest=f, index=HippoRAGWormholeIndex(f,damping_alpha=.65), ordinary_history_specs=None, fallback_pool_specs=[], open_code_agent_fallback=False, allow_pool_fallback=False)
        out[label] = rec
    return out


def dynamic_index_controls():
    f = load()
    ix = {'ppr':HippoRAGWormholeIndex(f,damping_alpha=.65), 'flat':FlatEqualMetadataIndex(f), 'compiled':CompiledRelationalCueTable(f)}
    tv = {'goal_kind':'open_workspace_mechanism_prediction', 'tolerance':.05, 'active_arity':2}
    cue = ['sig:internal_monod_shunt_specialization']
    node_id = 'stage1_W4_monod_shunt'
    def ask(label, qs=cue):
        vals = {}
        for name, obj in ix.items():
            r = obj.retrieve(qs,task_view=tv,top_k=100)
            nd = next((c['target_node'] for c in r['candidates'] if c['node_id'] == node_id),None)
            vals[name] = {'query_compute_cost':r['query_compute_cost'], 'refresh_cost':r.get('index_refresh_compute_cost',r.get('table_recompile_compute_cost',0)), 'refresh_count':getattr(obj,'refresh_count',getattr(obj,'recompile_count',0)), 'node_present':nd is not None, 'node_version':nd['version'] if nd else None, 'node_preconditions':nd['claim_contract']['preconditions'] if nd else None}
        return {'label':label, 'mutation_epoch':f.mutation_epoch, 'indices':vals}
    out = [ask('baseline')]
    cc = deepcopy(f.nodes[node_id]['claim_contract'])
    cc['preconditions'].append({'field':'b','op':'<=','value':1.6})
    f.empirical_revise_node(node_id,witness_episode_id='audit_index_revision',cause='protocol_or_scope_mismatch',updated_claim_contract=cc)
    f.revalidate_vertical_edges_after_revision(node_id,witness_episode_id='audit_index_rebind')
    out.append(ask('formal_node_revision'))
    out.append(ask('repeat_query_no_mutation'))
    wid = 'wh_arbor_to_vascular_2d_hub'
    f.refine_wormhole_with_near_miss(wid,near_miss_episode_id='audit_index_wh_revision',exclusion_predicate={'field':'audit_missing_property','op':'==','value':True})
    out.append(ask('formal_wormhole_revision'))
    for _ in range(3):
        f.record_utility_feedback(node_id,saved_cost=0.0,overhead_cost=1.0,succeeded=False)
    out.append(ask('utility_archival_after_three_negative_trials'))
    out.append(ask('repeat_archived_query_no_mutation'))
    return out


def initialize_part3_base(f):
    for hub, extra in (
        ('stage1_W3_murray_capped',('sig:cue_tubule_saturation_a','sig:cue_reuptake_saturation_b')),
        ('vascular_2d_hub',('sig:cue_photobioreactor_cap2d_a','sig:cue_allosteric_cap2d_b','sig:pos_arity_2')),
    ):
        n = f.nodes[hub]
        sigs = list(dict.fromkeys(n['probe_signatures'] + list(extra)))
        f.register_node(node_id=hub,tree_id=n['tree_id'],layer_role=n['layer_role'],task_view=n['task_view'],observation_map=n['observation_map'],scale_map=n['scale_map'],claim_contract=n['claim_contract'],probe_signatures=sigs,program_steps=n['program_steps'],source_cluster_ids=n['source_cluster_ids'])
        f.revalidate_vertical_edges_after_revision(hub,witness_episode_id='audit_part3_initial')
    f.revalidate_wormhole_after_revision('wh_arbor_to_vascular_2d_hub',witness_episode_id='audit_part3_initial')


def cost_analysis():
    rows = [json.loads(x) for x in (EV/'part3_m4_4_online_writeback_receipts.jsonl').read_text().splitlines()]
    sums = defaultdict(lambda:defaultdict(float))
    phases = defaultdict(lambda:defaultdict(float))
    per_step = defaultdict(dict)
    for r in rows:
        a = r['arm']
        for k in ('measurement_cost','compute_cost','composite_cost'):
            sums[a][k] += r['cost'][k]
        sums[a]['evaluated_candidates'] += r['evaluated_candidate_count']
        for p,rec in r['cost']['phase_breakdown'].items():
            phases[a][p] += rec.get('compute',0)
        per_step[r['stream_step']][a] = r
    on, off, comp = 'Hegel_LiveWriteBack_PPR_PoolOn','Hegel_LiveWriteBack_PPR_PoolOff','Compiled_LiveWriteBack_CueTable'
    paired = []
    for step, rs in sorted(per_step.items()):
        p, c = rs[on], rs[comp]
        paired.append({'step':step,'phase':p['stream_phase'],'PPR_candidates':p['evaluated_candidate_count'],'Compiled_candidates':c['evaluated_candidate_count'],'delta_compiled_minus_ppr_C':round(c['cost']['composite_cost']-p['cost']['composite_cost'],6),'PPR_selected':p['selected_node_id'],'Compiled_selected':c['selected_node_id'],'pool_on_off_candidate_equal':p['evaluated_candidate_count']==rs[off]['evaluated_candidate_count'],'pool_on_off_cost_equal':p['cost']==rs[off]['cost']})
    initial = load()
    initialize_part3_base(initial)
    ppr = HippoRAGWormholeIndex(initial,damping_alpha=.7)
    cu = CompiledRelationalCueTable(initial)
    ic = {'PPR_K':ppr.index_build_compute_cost,'Compiled_K':cu.compile_compute_cost,'PPR_edge_counts':ppr.edge_counts}
    corrected = {a:round(vals['composite_cost'] + .05*(cu.compile_compute_cost if a==comp else ppr.index_build_compute_cost),4) for a,vals in sums.items()}
    return {'row_count':len(rows),'total_costs':sums,'compute_phase_sums':phases,'paired_steps':paired,'initial_index_build_compute_not_in_receipts':ic,'with_initial_build_stream_totals':corrected,'raw_unrounded_K_advantage':round(sums[comp]['compute_cost']-sums[on]['compute_cost'],8),'raw_unrounded_C_advantage':round(.05*(sums[comp]['compute_cost']-sums[on]['compute_cost']),8),'sum_rounded_receipt_C_advantage':round(sums[comp]['composite_cost']-sums[on]['composite_cost'],8),'corrected_C_advantage':round(corrected[comp]-corrected[on],8)}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    write('archive_identity_checks.json', archive_checks())
    write('dynamic_index_controls.json', dynamic_index_controls())
    write('part3_cost_analysis.json', cost_analysis())
    write('certificate_controls.json', certificate_controls())
    print(json.dumps({'status':'complete','out_dir':str(OUT),'outputs':['archive_identity_checks.json','dynamic_index_controls.json','part3_cost_analysis.json','certificate_controls.json']}))
