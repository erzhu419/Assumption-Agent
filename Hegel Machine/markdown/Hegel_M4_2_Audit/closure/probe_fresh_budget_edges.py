from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import json
import hegel_machine.cognition.m4_multiscale_renormalization as m
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
OUT=Path(__file__).resolve().parent

def run(t,f,n,label):
    f=deepcopy(f)
    for nd in f.nodes.values():
        for s in nd.get('program_steps',[]):
            if s['step']=='subprocess_isolated_confirm':s['fresh_points']=n
    calls=[];original=m.MultiScaleWorkspaceTaskLab.acquire_channel
    def acq(self,pts,*,channel,phase):
        calls.append({'phase':phase,'channel':channel,'points':deepcopy(pts)})
        return original(self,pts,channel=channel,phase=phase)
    with patch.object(m.MultiScaleWorkspaceTaskLab,'acquire_channel',acq):
        result=m.evaluate_m4_target_under_arm(t,arm_name='Hegel_M4_MultiScale_HippoWalk',forest=f,index=HippoRAGWormholeIndex(f,damping_alpha=.65,wormhole_equivalence_pooling=True,scale_aware=True),stage1_gate=None,work_dir=OUT/'fresh_budget_workspaces'/label)
    return {'requested_fresh_per_confirm':n,'available_confirm_points':len(t.confirm_points),'result':result,'acquisitions':calls}
if __name__=='__main__':
    f,_=m.build_m4_multiscale_forest(cluster_seed=101,extra_distractor_count=6)
    ts,_=m.build_m4_cluster_targets(cluster_idx=0)
    r={'legal_three_only_two':run(ts[0],f,3,'requested_3'),'walk_parent_exhaustion':run(ts[10],f,2,'parent_exhaustion')}
    (OUT/'fresh_budget_results.json').write_text(json.dumps(r,indent=2))
    for k,v in r.items():print(k,json.dumps({'strict_pass':v['result']['strict_pass'],'confirmed':v['result'].get('confirmed_ok'),'acquisitions':v['acquisitions']}))
