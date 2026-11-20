"""Finite offline M4.2 contract controls; genuine committed artifact, no model calls."""
from pathlib import Path
from copy import deepcopy
import json, math, shutil, hashlib
from hegel_machine.cognition.m4_compositional_discovery import load_verified_authored_method, predict_with_frozen_instance
from hegel_machine.cognition.m4_multiscale_renormalization import build_m4_multiscale_forest, build_m4_cluster_targets, _validate_and_interpret_m4_program_steps
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use
OUT=Path(__file__).resolve().parent
REPO=Path('/workspace/scratch/3f14ad2da0ed/repo_m42_fb4b23aaf/Hegel Machine')
EV=REPO/'docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/persisted_authored_workspaces/persisted_stage1_authored_workspaces'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(a, rows):
    res=predict_with_frozen_instance(a,[r['features'] for r in rows])
    return {'rmse':math.sqrt(sum((p-r['target'])**2 for p,r in zip(res['predictions'],rows))/len(rows)),**res}
def auth(n,scope):
    return authorize_abstraction_use(requested_tier='bounded_trial',task_view={'goal_kind':'open_workspace_mechanism_prediction','tolerance':.1},target_features={'audit_scope':scope},claim_contract=n['claim_contract'],scale_map=n['scale_map'])
def verdict(n,q):
    branches=n['claim_contract'].get('disjunctive_branches')or[n['claim_contract']]
    return 'ACC' if any(all(q.get(p['field'])==p['value'] for p in b['preconditions'])for b in branches)else 'UNK'
def setup_nodes(forest):
    f=deepcopy(forest)
    a,b='aq_macro_river_tree','dist_node_00'
    for nid,scope in[(a,'A'),(b,'B')]:
        n=f.nodes[nid]
        n['observation_map']=deepcopy(f.nodes[a]['observation_map'])
        n['scale_map']=deepcopy(f.nodes[a]['scale_map'])
        n['claim_contract']['conclusion_or_action']=deepcopy(f.nodes[a]['claim_contract']['conclusion_or_action'])
        n['claim_contract']['preconditions']=[{'field':'audit_scope','op':'==','value':scope}]
        n['claim_contract']['permission_tier']='bounded_trial'
    return f,a,b

def main():
    result={'commit':'fb4b23aaf802c20b4e2d0018436abab73b0cca06'}
    dst=OUT/'param_snapshot_copy'
    shutil.copytree(EV/'stage1_W1_haldane_shunt',dst,dirs_exist_ok=True)
    p=dst/'authored_method.json'
    orig=json.loads(p.read_text()); rows=orig['host_confirmation_bundle']['confirm_rows']
    good=load_verified_authored_method(p,expected_ast_digest=orig['ast_skeleton_digest'],require_external_manifest=True)
    untouched={n:sha(dst/n)for n in['confirmed_manifest.json','confirmed_fitted_params.json','confirmed_observations.json','authored_model.py']}
    bad=deepcopy(orig);bad['fitted_params']['p0']*=3
    p.write_text(json.dumps(bad,indent=2))
    badload=load_verified_authored_method(p,expected_ast_digest=orig['ast_skeleton_digest'],require_external_manifest=True)
    result['fitted_state']={'baseline_accepted':good is not None,'baseline_frozen':rmse(good,rows),'modified_accepted':badload is not None,'modified_frozen':rmse(badload,rows)if badload else None,'changed_only_top_level_fields':[k for k in orig if orig[k]!=bad[k]],'mutated_param':'p0 * 3','source_and_confirmation_bundle_unchanged':all(orig[k]==bad[k]for k in['model_sha256','ast_skeleton_digest','authored_model_source','host_confirmation_bundle','confirmation_digest']),'external_evidence_unchanged':{n:sha(dst/n)==h for n,h in untouched.items()},'tolerance':orig['host_confirmation_bundle']['tolerance']}
    # Fresh genuine bundle node-binding control; avoids legacy-artifact precondition of original probe.
    p.write_text(json.dumps(orig,indent=2))
    other=json.loads((EV/'stage1_W2_osc_shunt_4p'/'authored_method.json').read_text())
    slot=OUT/'binding_control';shutil.copytree(EV/'stage1_W2_osc_shunt_4p',slot,dirs_exist_ok=True)
    expected=orig['ast_skeleton_digest']
    result['fresh_binding']={'original_expected_accepted':load_verified_authored_method(p,expected_ast_digest=expected,require_external_manifest=True)is not None,'other_self_consistent_accepted':load_verified_authored_method(slot/'authored_method.json',expected_ast_digest=other['ast_skeleton_digest'],require_external_manifest=True)is not None,'other_with_original_expected_rejected':load_verified_authored_method(slot/'authored_method.json',expected_ast_digest=expected,require_external_manifest=True)is None}
    forest,_=build_m4_multiscale_forest(cluster_seed=101,extra_distractor_count=6)
    f,a,b=setup_nodes(forest);f.nodes[b]['claim_contract']['permission_tier']='retrieve_candidate'
    before=auth(f.nodes[b],'B')
    rep=f.compress_compatible_nodes(keeper_node_id=a,redundant_node_id=b,evaluation_queries=[{'audit_scope':'A'},{'audit_scope':'B'},{'audit_scope':'C'}],evaluator_fn=verdict)
    result['or_permission']={'merge_accepted':rep['accepted'],'B_before':before,'B_after':auth(f.nodes[a],'B'),'top_permission':f.nodes[a]['claim_contract']['permission_tier'],'branch_permissions':[br['permission_tier']for br in f.nodes[a]['claim_contract']['disjunctive_branches']]}
    f,a,b=setup_nodes(forest);f.nodes[b]['claim_contract']['conclusion_or_action']=deepcopy(forest.nodes['bio_vascular_macro_capped']['claim_contract']['conclusion_or_action']);f.nodes[b]['program_steps']=deepcopy(forest.nodes['bio_vascular_macro_capped']['program_steps'])
    rep=f.compress_compatible_nodes(keeper_node_id=a,redundant_node_id=b,evaluation_queries=[{'audit_scope':'A'},{'audit_scope':'B'},{'audit_scope':'C'}],evaluator_fn=verdict)
    result['or_action']={'merge_accepted':rep['accepted'],'B_authorized':auth(f.nodes[a],'B'),'top_action':f.nodes[a]['claim_contract']['conclusion_or_action'],'branch_actions':[br['conclusion_or_action']for br in f.nodes[a]['claim_contract']['disjunctive_branches']],'gate_returns_selected_branch_or_action':any(k in auth(f.nodes[a],'B')for k in['selected_branch','selected_action','conclusion_or_action'])}
    node=forest.nodes[a];prog=deepcopy(node['program_steps'])
    result['program_order']={'canonical_plan':_validate_and_interpret_m4_program_steps(prog,node=node,target_arity=1),'reversed_plan':_validate_and_interpret_m4_program_steps(list(reversed(prog)),node=node,target_arity=1)}
    (OUT/'contract_controls_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
