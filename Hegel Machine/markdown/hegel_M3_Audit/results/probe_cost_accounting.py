"""Matched synthesis inputs and logged-vs-independent bridge meter, no source edits."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from statistics import mean

REPO=Path(os.environ.get('HEGEL_REPO','/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine'))
OUT=Path(os.environ.get('AUDIT_OUTPUT_DIR',Path(__file__).resolve().parent))
sys.path.insert(0,str(REPO/'src'))
import hegel_machine.cognition.m3_scale_universe as m

orig=REPO/'docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119'
source=[json.loads(x) for x in (orig/'source_episodes.jsonl').read_text().splitlines()]
targets=[json.loads(x) for x in (orig/'target_episodes.jsonl').read_text().splitlines()]
suite=m.make_m3_large_scale_suite(7027101)
_,name,(fac,ev)=suite['targets'][0]
lab=fac();work=OUT/'cost_probe_workspaces';work.mkdir(parents=True,exist_ok=True)
lens=m._estimate_observation_lens(lab,work/'common_pilot')
lab.acquire(m.build_m3_pilot_points(lab),workspace=work/'common_pilot',phase='presolve')
arity=len(m._active_and_fixed_features(lab)[0])
priors=sorted({r['selected_schema'] for r in source if r['cluster_id']=='m3_eval_01' and r['arm']=='O_strong_AllValidatedPriors' and r['strict_pass'] and r['active_arity']==arity})
matched={}
for kind,allowed in [('all_validated_priors',priors),('unrestricted_same_script',None)]:
    clone=deepcopy(lab);before=clone.meter.compute_spent
    fit=m.run_m3_workspace_synthesizer(work/kind,clone,observation_lens=lens,allowed_schemas=allowed,output_model_path='model.py')
    matched[kind]={'allowed_schemas':allowed,'selected_schema':fit.get('selected_schema'),'train_rmse':fit.get('train_rmse'),
                   'val_rmse':fit.get('val_rmse'),'model_sha256':hashlib.sha256((work/kind/'model.py').read_bytes()).hexdigest(),
                   'charged_compute':clone.meter.compute_spent-before,'actual_wall_seconds':fit.get('measured_wall_seconds'),
                   'fit_result_keys':list(fit)}
matched['same_model_bytes']=matched['all_validated_priors']['model_sha256']==matched['unrestricted_same_script']['model_sha256']
matched['candidate_count']=len(priors)

# Record cost of precisely the existing extra bridge fit while running a single
# original source formation. The wrapper only reads the independent lab meter.
bridge=[];original=m.run_m3_workspace_synthesizer
def observe_bridge(workspace,lab,**kwargs):
    value=original(workspace,lab,**kwargs)
    if lab.task_id=='m3_bridge_fit':
        bridge.append({'lab_task_id':lab.task_id,'meter':lab.meter.snapshot(),'measured_wall_seconds':lab.measured_python_wall_seconds,
                       'train_rmse':value.get('train_rmse'),'allowed_schemas':kwargs.get('allowed_schemas')})
    return value
m.run_m3_workspace_synthesizer=observe_bridge
researcher=m.M3ForestResearcher()
logs=researcher.form_and_revise_36_episodes('cost_probe_eval01',suite,work/'source_formation')
m.run_m3_workspace_synthesizer=original
bridge_result={'independent_bridge_meter':bridge,'returned_source_log_count':len(logs),'returned_source_phases':{},
               'returned_log_composite_total':sum(r['costs']['composite_cost'] for r in logs),
               'returned_log_has_bridge_episode':any('bridge' in r.get('episode_name','') for r in logs),
               'omitted_bridge_composite':sum(x['meter']['composite_cost'] for x in bridge),
               'scope':'One original source formation; same extra synthesis occurs once in each of 2 DEV and 4 EVAL forest formations.'}
for r in logs:bridge_result['returned_source_phases'][r['phase']]=bridge_result['returned_source_phases'].get(r['phase'],0)+1

# Derived re-pricing: cap first same-arity all-prior call at unrestricted full
# search's existing 48-unit fee, without changing any actions, measurements,
# confirmations, outcomes, source costs, or other target cost components.
by_cluster={}
for cid in sorted({r['cluster_id'] for r in targets}):
    rows=[r for r in targets if r['cluster_id']==cid and r['arm']=='O_strong_AllValidatedPriors']
    src=[r for r in source if r['cluster_id']==cid and r['arm']=='O_strong_AllValidatedPriors' and r['strict_pass']]
    by_ar={ar:sorted({r['selected_schema'] for r in src if r['active_arity']==ar}) for ar in {r['active_arity'] for r in src}}
    # Published source task arities are read from target public feature schema;
    # replay does not run when deriving this sensitivity.
    suite_c=m.make_m3_large_scale_suite(dict([('m3_eval_01',7027101),('m3_eval_02',7027202),('m3_eval_03',7027303),('m3_eval_04',7027404)])[cid])
    arity_by_name={nm:len(m._active_and_fixed_features(factory())[0]) for _,nm,(factory,_) in suite_c['targets']}
    saved_compute=sum(max(0,10*len(by_ar[arity_by_name[r['target_name']]])-48) for r in rows)
    hrows=[r for r in targets if r['cluster_id']==cid and r['arm']=='Hegel_M3_HippoPPR']
    original_delta=sum(r['costs']['composite_cost'] for r in hrows)-sum(r['costs']['composite_cost'] for r in rows)
    by_cluster[cid]={'priors_by_arity':by_ar,'ordinary_saved_proxy_compute':saved_compute,
                    'original_hegel_minus_ordinary_target_composite':original_delta,
                    'repriced_hegel_minus_ordinary_target_composite':original_delta+.05*saved_compute,
                    'hegel_minus_ordinary_measurement_unchanged':sum(r['costs']['measurement_cost'] for r in hrows)-sum(r['costs']['measurement_cost'] for r in rows)}
result={'fixed_commit':'612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8','matched_synthesis':matched,'bridge_accounting':bridge_result,
        'fee_cap_sensitivity':{'description':'Post-hoc accounting sensitivity; first prior-list fit capped at original full-search fee48; no scientific behavior altered.',
                              'per_cluster':by_cluster,
                              'mean_hegel_minus_ordinary_target_composite':mean(x['repriced_hegel_minus_ordinary_target_composite'] for x in by_cluster.values()),
                              'mean_ordinary_saved_compute':mean(x['ordinary_saved_proxy_compute'] for x in by_cluster.values())}}
(OUT/'cost_accounting_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
