"""Use archived source instance to test real authorization after observed S01 protocol edit.
No fitting, new measurements or source edits; S01 state from probe_s01_failure.py.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import load_location_independent_forest
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import ForestScopeDispatcher
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use
BASE=Path(os.environ.get('HEGEL_AUDIT_REPO','/workspace/scratch/3f14ad2da0ed/repo_m44_b965f022b/Hegel Machine'))
OUT=Path(os.environ.get('HEGEL_AUDIT_OUT','/workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online'))
EVID=BASE/'docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120'
state=json.loads((OUT/'s01_failed_control.json').read_text())
recs=[]
for phase in ('before','after'):
 f=load_location_independent_forest(EVID/'relocatable_forest_snapshots/post_stage2_forest_snapshot.json',EVID)
 nd=deepcopy(state[phase]); f.nodes[nd['node_id']]=nd
 for bleed in (1.5,1.7):
  feat={'s':4.0,'b':bleed}
  meta={'active_arity':2,'haldane_turnover_shunt_regime':False,'secondary_bleed_active':True}
  gate=authorize_abstraction_use(requested_tier='bounded_trial',task_view=nd['task_view'],target_features={**feat,**meta},claim_contract=nd['claim_contract'],scale_map=nd.get('scale_map'))
  rec=ForestScopeDispatcher(f).dispatch_predict(query_id=f'{phase}_b_{bleed}',candidate_tree_id='tree_monod_reactor',features=feat,target_metadata=meta)
  recs.append({'phase':phase,'features':feat,'gate_granted':gate['granted'],'gate_reason':gate['reason'],'dispatch':rec})
result={'control':'Actual authorizer and zero-fit dispatcher before/after S01 protocol-only revision', 'observations_acquired':0,'model_api_calls':0,'rows':recs}
(OUT/'s01_scope_control.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
print(json.dumps(result,ensure_ascii=False,indent=2))
