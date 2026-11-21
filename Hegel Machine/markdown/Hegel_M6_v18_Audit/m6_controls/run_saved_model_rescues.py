"""Reuse paid model artifacts, without LLM calls or source changes."""
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import json,sys
ROOT=(Path(os.environ.get('HEGEL_M6_REPO', str(Path(__file__).resolve().parents[2] / 'repo_m6_b1d0c3e99'))) / 'Hegel Machine')
sys.path.insert(0,str(ROOT/'src'))
from hegel_machine.cognition.m6_1_zero_leakage_ast_causal_benchmark import build_m6_1_clean_k40_forest,consolidate_wave1_via_ast_wormholes
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves
from hegel_machine.cognition.m5_live_gpt_and_provenance import _execute_authored_python_in_isolated_subprocess
OUT=Path(__file__).resolve().parent
waves=build_m6_causal_24_waves()
root=ROOT/'docs/evidence/m6_1_zero_leakage_ast_causal_192ep_20261121/workspaces_and_dags/wave_1'
def evaluate(code,train,ds):
    with TemporaryDirectory() as td:
        r=_execute_authored_python_in_isolated_subprocess(Path(td),step_idx=1,python_code=code,train_rows=train,confirm_rows=ds.confirm_rows,heldout_rows=ds.heldout_rows)
    return {k:v for k,v in r.items() if k in ('ok','train_rmse','confirm_rmse','heldout_rmse','error')}
results={}
for taskid,step in [('W1R2__C05_ColliderShielded_DirectX',1),('W1R1__C03_ColliderBerkson_tree_C03_S1_catalog',2)]:
    ds=next(d for d in waves['wave_1'] if d.task_id==taskid)
    code=(root/('gpt-6-luna__Hegel_M6_1_ZeroLeakage_LiveGPT__'+taskid)/f'model_step_{step:02d}.py').read_text()
    train=list(ds.pilot_rows)
    if '__C03' in taskid:train+=ds.probe_pool['probe_unselected_S0_regime']['rows']
    adapted=code+'\n_fit_original = fit\ndef fit(rows):\n    global predict\n    fitted = _fit_original(rows)\n    if callable(fitted): predict = fitted\n'
    results[taskid]={'original':evaluate(code,train,ds),'fit_callable_adapted':evaluate(adapted,train,ds)}
(OUT/'fit_contract_control.json').write_text(json.dumps(results,indent=2))
rs=[]
for path in root.glob('*/episode_receipt.json'):
    r=json.loads(path.read_text())
    if r['model']=='gpt-6-luna' and r['arm']=='Hegel_M6_1_ZeroLeakage_LiveGPT' and '__C02_' in r['task_id']:rs.append(r)
outputs={}
for label,reverse in [('R1_first',False),('R2_first',True)]:
    f,a=build_m6_1_clean_k40_forest()
    c=consolidate_wave1_via_ast_wormholes(wave1_receipts=sorted(rs,key=lambda r:r['task_id'],reverse=reverse),arm='Hegel_M6_1_ZeroLeakage_LiveGPT',forest=f,node_archive=a)
    outputs[label]={'events':c['events']}
    code=a['node_C02_TreeM2_analog'].get('confirmed_python_reference')
    if code:
        for ds in waves['wave_2']:
            if '__C02_' in ds.task_id:outputs[label][ds.task_id]=evaluate(code,ds.pilot_rows,ds)
(OUT/'luna_c02_order_control.json').write_text(json.dumps(outputs,indent=2))
print(json.dumps({'fit_contract_control':results,'luna_c02_order_control':outputs},indent=2))
