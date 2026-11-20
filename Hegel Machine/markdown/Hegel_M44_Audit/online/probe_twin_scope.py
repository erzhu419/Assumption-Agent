"""Reuse an actual S02 artifact produced by the official test/study. No rediscovery.
Set HEGEL_ONLINE_W6_ARTIFACT to online_writebacks/online_W6_michaelis_cap_1d/authored_method.json.
This tests alias scope, not the successful target-refitting path used in the official 20-step stream.
"""
import json
import os
from pathlib import Path
from random import Random
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import _write_back_discovered_mechanism_to_forest
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import ForestScopeDispatcher
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest
OUT=Path(os.environ.get('HEGEL_AUDIT_OUT','/workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online'))
source=Path(os.environ['HEGEL_ONLINE_W6_ARTIFACT'])
art=json.loads(source.read_text())
assert art['task_id']=='M44_S02_Discover_W6_Haldane1D', art['task_id']
assert art['confirmed_ok'] is True
bundle=OUT/'twin_scope_bundle'
f=PhilosophicalForest(artifact_root=str(bundle.resolve()))
_write_back_discovered_mechanism_to_forest(f,ep_res={'authored_artifact':art},artifact_abs_dir=bundle/'persisted_authored_workspaces/online_writebacks/online_W6_michaelis_cap_1d',bundle_root=bundle,primary_node_id='online_W6_michaelis_cap_1d',primary_tree_id='tree_online_tubule_cap',operator_tag='op_online_W6_haldane_1d',arity=1,primary_cues=['sig:cue_tubule_saturation_a'],preconditions=[{'field':'active_arity','op':'==','value':1}],weaker_parent_id=None,twin_node_id='online_W6_reuptake_twin_1d',twin_tree_id='tree_online_reuptake_cap',twin_cues=['sig:cue_reuptake_saturation_b','sig:pos_arity_1','sig:neg_arity_1'],wormhole_id='wh_online_W6_tubule_reuptake',tolerance=.045)
x=3.2  # actual S04 heldout coordinate from the official stream definition
source_truth=14.5*x/(1.6+x+.18*x*x)
rng=Random(20361120)
for _ in range(2):
 vm=round(rng.uniform(13.2,15.8),3);km=round(rng.uniform(1.4,1.9),3);ki=round(rng.uniform(.15,.22),4)
s04_truth=vm*x/(km+x+ki*x*x)
checks=[]
for label,tree,truth in [('primary_original_source','tree_online_tubule_cap',source_truth),('twin_as_source_alias','tree_online_reuptake_cap',source_truth),('twin_with_actual_unseen_S04_instance','tree_online_reuptake_cap',s04_truth)]:
 checks.append({'case':label,'truth_context':'S04' if label.endswith('instance') else 'S02 source', 'receipt':ForestScopeDispatcher(f).dispatch_predict(query_id=label,candidate_tree_id=tree,features={'x_load':x},target_metadata={'active_arity':1},ground_truth=truth)})
primary=f.nodes['online_W6_michaelis_cap_1d'];twin=f.nodes['online_W6_reuptake_twin_1d']
out={'control':'Official source artifact and actual production twin registration, followed by zero-fit dispatch','source_artifact_path':str(source),'source_task_id':art['task_id'],'source_surface_domain':art['surface_domain'],'source_instance_digest':art['confirmed_instance_digest'],'source_confirmation_count':len(art['host_confirmation_bundle']['confirm_rows']),'source_training_prefix_count':len(art['host_confirmation_bundle']['training_prefix_rows']),'source_composed_history':art['composed_history'],'source_diagnostic_report':art['diagnostic_report'],'primary_and_twin_action_identical':primary['claim_contract']['conclusion_or_action']==twin['claim_contract']['conclusion_or_action'],'twin_preconditions':twin['claim_contract']['preconditions'],'twin_evidence_status':twin['claim_contract']['evidence_status'],'twin_permission_tier':twin['claim_contract']['permission_tier'],'wormhole':f.horizontal_wormhole_edges['wh_online_W6_tubule_reuptake'],'S04_params':{'vm':vm,'km':km,'ki':ki},'checks':checks,'new_fits':0,'new_observations':0,'model_api_calls':0,'interpretation':'S04 official evaluation refits and confirms successfully; this control only shows that a cloned source instance must not be interpreted as a separately confirmed physical instance in the other domain.'}
(OUT/'twin_scope_control.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True))
print(json.dumps({k:v for k,v in out.items() if k not in ['source_composed_history','wormhole']},ensure_ascii=False,indent=2))
