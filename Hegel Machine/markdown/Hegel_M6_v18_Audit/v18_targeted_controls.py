"""Read-only local counterfactual controls for fixed commit 922b95e19.

Does not call any LLM/provider or modify the repository. Repairs below execute
only in-memory copies of already committed authored code, fitted to the same
committed training rows. Generated confirmation/heldout targets are only used
for retrospective diagnostics, never for fitting.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import random
import sys
from unittest.mock import patch

ROOT = (Path(os.environ.get('HEGEL_V18_REPO', str(Path(__file__).resolve().parent.parent / 'repo_v18_922b95e19'))) / 'Hegel Machine')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from hegel_machine.cognition import m5_live_gpt_and_provenance as live
from hegel_machine.cognition import m5_temporal_multiscale_and_base_subtree as det
from hegel_machine.cognition.m5_v18_2_sovereign_closure import build_v18_2_live_challenge_tasks
from hegel_machine.cognition.m5_v18_3_continual_and_complex_live import apply_online_wave_consolidation
from hegel_machine.cognition.m5_v18_4_hegel_discovery_240 import (
    build_live_k45_forest_and_archive, build_hegel_discovery_240_waves,
    apply_v18_4_online_wave_consolidation,
)
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex
from hegel_machine.cognition.wormhole_forest import validate_consistent_joint_subtree_binding

results = {'commit':'922b95e194d17c9259f1993706440e0602a43db9',
           'scope':'Read-only targeted counterfactuals; no paid provider calls; not a replacement benchmark.'}
def checkpoint(name, value):
    results[name] = value
    (OUT/'v18_targeted_controls_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(name, json.dumps(value, ensure_ascii=False)[:1600], flush=True)
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()

# 1. Holding evaluation metadata fixed, change/remove successful programs.
records = [dict(subfamily=fam, task_id='opaque_task_'+str(i), status='confirmed_ok', strict_pass=True)
    for i,fam in enumerate(['F07_CriticalThreshold_GrayZone_Avalanche',
                           'F11_SpatioTemporal_SiphonMemory_5P','F12_SinkCoupled_DualHorizon_5P'])]
states=[]
for variant in ('no_code','different_code','renamed_family'):
    f,p,a=build_live_k45_forest_and_archive()
    rs=deepcopy(records)
    if variant=='different_code':
        for r in rs:r.update(python_code='def predict(row): return -999', fitted_parameters={'tau':999})
    if variant=='renamed_family':
        for i,r in enumerate(rs):r['subfamily']='anonymous_family_'+str(i)
    receipt=apply_v18_4_online_wave_consolidation(forest=f,node_archive=a,wave_receipts=rs)
    learned={k:v for k,v in a.items() if k.startswith(('node_O_','node_P_','node_S_'))}
    states.append(dict(variant=variant,receipt=receipt,learned_notes=learned,learned_digest=digest(learned)))
checkpoint('metadata_only_consolidation_v18_4', states)
states=[]
for variant in ('semantic_task_id','opaque_task_id'):
    f,p,a=live._build_live_k15_forest_and_archive()
    rs=[dict(task_id=tid,status='confirmed_ok',strict_pass=True) for tid in
        (['CX01_SpatioTemporal_SiphonMemory','CX02_CriticalThreshold_AvalancheSublation']
         if variant=='semantic_task_id' else ['opaque_01','opaque_02'])]
    receipt=apply_online_wave_consolidation(forest=f,node_archive=a,wave_receipts=rs)
    states.append(dict(variant=variant,receipt=receipt))
checkpoint('task_id_only_consolidation_v18_3', states)

# 2. Omitting or supplying unrelated causal edges, with all roles fixed.
f,p,a=build_live_k45_forest_and_archive()
sub=f.shared_base_subtrees['sub_2layer_siphon_base']
binding={'role_bindings':{'u_drive':'u_now','x_pool':'x_now','s_grad':'s_grad','y_out':'y'},
         'satisfied_shared_node_ids':['C1_base_capacity_field','C2_adjacent_high_gradient_sink'],
         'normalized_tau_ratio':1.0}
cases={}
for tag,extra in [('omitted',{}),('explicit_empty',{'edge_bindings':[]}),
                 ('unrelated_edge',{'edge_bindings':[{'from_slot':'irrelevant','to_slot':'other',
                                                    'target_from':'a','target_to':'b'}]})]:
    cases[tag]=validate_consistent_joint_subtree_binding(sub,{**binding,**extra})
checkpoint('joint_binding_edge_coverage',cases)

# 3. Live caller ignores a correctly denied numeric permission result.
waves=build_hegel_discovery_240_waves(20261121)
tasks={t.task_id:t for ws in waves.values() for t in ws}
ds=next(t for t in waves['wave_1'] if t.subfamily=='F05_TwoLayer_ConjunctiveSiphon_Subtree')
def make_brief(ff):
    led=live.DecisionProvenanceLedger(run_id='audit',task_id=ds.task_id,arm='Hegel_M5_LiveGPT',model='no_model',provider_backend='none')
    d0=led.record_decision(decision_type='D0',parent_decision_id=None,evidence_inputs={},candidates_evaluated=[],chosen_option={},rejection_reasons={},rationale_summary='Audit only')
    _,brief,_=live._build_arm_system_and_initial_brief(ds,'Hegel_M5_LiveGPT',init_rows=ds.pilot_rows,forest=ff,
        priors_bundle=p,node_archive=a,hippo_idx=HippoRAGWormholeIndex(ff),flat_idx=FlatEqualMetadataIndex(ff),ledger=led,parent_id=d0)
    return brief['shared_base_subtree_guidance']
normal=make_brief(f)
denied = {**validate_consistent_joint_subtree_binding(sub,binding), 'numeric_warm_start_authorized':False,
          'activation_tier':'retrieve_candidate_only', 'reason':'audit_forced_denied_numeric_permission'}
with patch.object(live,'validate_consistent_joint_subtree_binding',return_value=denied):
    rejected=make_brief(f)
checkpoint('live_numeric_permission_denial',dict(control='Inject a denied binding result; all observations and source content unchanged',
    denied_result=denied,normal_guidance=normal,denied_guidance=rejected,guidance_unchanged=normal==rejected))

# 4. Forced HEC must buy zero; compare the true pipeline and independent traced copy.
f,p=det._build_seed_forest_and_priors(); hi=HippoRAGWormholeIndex(f); fi=FlatEqualMetadataIndex(f)
ds=next(t for t in det._generate_m5_benchmark_tasks(101) if t.subfamily=='C_two_layer_conjunctive_subtree')
hec={'action':'hec_decision_region_stop','selected_probe_id':None,'compute_cost':0.0,'policy':'audit_forced_hec','voi_receipt':{}}
with patch.object(det,'select_m5_discriminating_probe',return_value=hec):
    rg=det._run_genuine_solver_pipeline(ds,'Hegel_M5_Full_TemporalSubtree_Forest',forest=f,priors_bundle=p,hippo_idx=hi,flat_idx=fi,rng=random.Random(12345))
with patch.object(live,'select_m5_discriminating_probe',return_value=hec):
    rt,ch=live.run_traced_deterministic_m5_episode(ds,'Hegel_M5_Full_TemporalSubtree_Forest',forest=f,priors_bundle=p,hippo_idx=hi,flat_idx=fi,rng=random.Random(12345))
checkpoint('traced_vs_genuine_hec_stop',dict(task_id=ds.task_id,genuine=rg,traced=rt,
    chain_verifier=live.verify_decision_provenance_chain(ch),
    traced_probe_records=[x for x in ch if x['decision_type'].startswith('D5')]))

# 5. Raw-prefix reader's spelling mismatch: internal history can be changed invisibly.
l08=build_v18_2_live_challenge_tasks()[1]
seen_lengths=[]; orig_extractor=det.extract_causal_temporal_regimes
def spy(prefix,*args,**kwargs):
    seen_lengths.append(len(prefix)); return orig_extractor(prefix,*args,**kwargs)
with patch.object(det,'extract_causal_temporal_regimes',side_effect=spy):
    fit=det._fit_candidate_model_class('M4_MoriZwanzigMemoryKernel',l08.pilot_rows,
        tau_scale=l08.tau_scale,warm_prior=None,tolerance=l08.tolerance,rng=random.Random(1))
    row=deepcopy(l08.confirm_rows[0]); changed=deepcopy(row)
    for point in changed['raw_prefix'][1:-1]:point['x']+=100
    before=fit['predict_fn'](row);after=fit['predict_fn'](changed)
checkpoint('raw_prefix_middle_history_invariance',dict(task_id=l08.task_id,
    actual_history_length=len(row['raw_prefix']),extractor_prefix_lengths=sorted(set(seen_lengths)),
    original_prediction=before,changed_history_prediction=after,abs_difference=abs(before-after),
    changed_only='Interior raw_prefix x values +100; first and last anchors and all current row features fixed'))

# 6. Saved-code fit/predict ABI repair on the very same rows, no equation change.
evroot=ROOT/'docs/evidence/v18_4_hegel_discovery_240_20261121/workspaces_and_dags'
def run_code(code,payload,ds):
    ns={}; exec(compile(code,'<in-memory-committed-or-local-repair>','exec'),ns)
    pars=ns['fit'](payload['train_rows'])
    out={}
    for key,rs in [('train',payload['train_rows']),('confirm',ds.confirm_rows),('heldout',ds.heldout_rows)]:
        xx=[ns['predict']({k:v for k,v in r.items() if k!='y'}) for r in rs]
        out[key+'_rmse']=float(sum((float(pred)-float(row['y']))**2 for pred,row in zip(xx,rs))**0.5/len(rs)**0.5)
    out['fit_returned_parameters']=None if pars is None else [float(x) for x in pars]
    return out
tid='W1R3__F06_NarrowPilot_HeteroParam_WideEscape'
ws=evroot/'wave_1'/('gpt-6-luna__Hegel_M5_LiveGPT__'+tid)
code=(ws/'model_step_02.py').read_text();payload=json.loads((ws/'eval_payload_02.json').read_text())
patched=code.replace('def fit(train_rows):','_PARAMS = None\n\ndef fit(train_rows):\n    global _PARAMS').replace('    return result.x','    _PARAMS = result.x\n    return result.x').replace("vmax, rho, c_loc = fit([dict(row, y=row.get('y', 0.0))])","vmax, rho, c_loc = _PARAMS")
assert patched!=code
checkpoint('saved_code_abi_repair',dict(task_id=tid,source=str(ws/'model_step_02.py'),
    original=run_code(code,payload,tasks[tid]), repaired=run_code(patched,payload,tasks[tid]),
    change='Only save fit result in module variable and reuse in predict; equation, data, bounds, starts unchanged'))

# 7. Saved-code sign-bound repair, same law and all observations.
tid='W2R1__F10_LSR_Transformed_BarrierSiphon'
ws=evroot/'wave_2'/('gpt-6-luna__Hegel_M5_LiveGPT__'+tid)
code=(ws/'model_step_01.py').read_text();payload=json.loads((ws/'eval_payload_01.json').read_text())
patched=code.replace('bounds=([0.0, 0.0, 0.05, 0.0]', 'bounds=([0.0, 0.0, 0.05, -2.0]')
assert patched!=code
checkpoint('saved_code_signed_branch_repair',dict(task_id=tid,source=str(ws/'model_step_01.py'),
    original=run_code(code,payload,tasks[tid]),repaired=run_code(patched,payload,tasks[tid]),
    change='Only c_sqrt lower bound 0 -> -2; no new rows, no changes to law, starts, tolerance or other bounds'))

print('COMPLETE',flush=True)
