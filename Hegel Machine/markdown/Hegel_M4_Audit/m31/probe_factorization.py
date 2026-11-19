"""Focused one-seed controls: fill missing PPR alpha=0/no-pool cell.

Reuse fixed V17.1 suite, source formation, published DEV gate, public pilots,
same scorer/metadata/contracts/execution budgets. No repository edits/API calls.
This is a post-hoc diagnostic, not another preregistered result.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

import hegel_machine.cognition.m3_scale_universe as m3
from hegel_machine.cognition.m3_1_walk_factorization import M3_1FactorizedResearcher, make_m3_1_factorized_walk_suite, build_m3_1_pilot_points, extract_m3_1_observable_signatures
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_python_revision import run_open_revision_chain_experiment
from hegel_machine.cognition.wormhole_forest import TwoStageWormholeGate

ROOT = Path(__file__).resolve().parent
REPO = Path(m3.__file__).resolve().parents[3]
SEED = 8_028_101
COMMIT = '11961f702739ab272661ff59b77fb5bc8e3589b8'


def tag(n):
    return n['claim_contract']['conclusion_or_action']['schema_tag']


def make_gate():
    g = TwoStageWormholeGate(margin=1., uncovered_exploration_budget_calls=12)
    d = json.loads((REPO/'docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/dev_gate_calibration.json').read_text())
    g.paired_dev_records = deepcopy(d['paired_dev_records'])
    return g


def check_k_inputs():
    out = []
    for k in (20, 36, 52):
        suite = make_m3_1_factorized_walk_suite(SEED, target_k=k)
        targets = []
        for cat, name, (fac, _) in suite['targets']:
            lab = fac()
            sample_points = [{f: lo+q*(hi-lo) for f,(lo,hi) in lab.feature_bounds.items()} for q in (.04, .22, .66, .88)]
            targets.append({'category':cat, 'name':name, 'public_view':lab.task_view, 'sample_oracle_values':[lab._base_oracle_fn(p) for p in sample_points], 'noise_seed':lab._noise_seed})
        payload = json.dumps(targets, sort_keys=True).encode()
        out.append({'K':k,'source_count':len(suite['sources']), 'near_miss_count':len(suite['near_misses']), 'source_names':[x[0] for x in suite['sources']], 'target_digest':hashlib.sha256(payload).hexdigest(), 'target_count':len(targets)})
    return {'settings':out,'all_target_inputs_identical':len({x['target_digest'] for x in out})==1}


def pricing_control():
    suite = m3.make_m3_large_scale_suite(7_027_101)
    _, name, (fac, _) = suite['targets'][0]
    lab = fac()
    lens = m3._estimate_observation_lens(lab,ROOT/'pricing/common')
    lab.acquire(m3.build_m3_pilot_points(lab),workspace=ROOT/'pricing/common',phase='presolve')
    full_tags = list(dict.fromkeys(re.findall(r'allow\("([^"]+)"\)', m3._M3_SYNTHESIZER_SCRIPT)))
    out = []
    for label, allowed in [('singleton',['liebig_capped_min_2d']), ('explicit_complete_grammar',full_tags), ('unrestricted',None)]:
        l = deepcopy(lab)
        before = l.meter.compute_spent
        fit = m3.run_m3_workspace_synthesizer(ROOT/'pricing'/label,l,observation_lens=lens,allowed_schemas=allowed)
        out.append({'label':label,'evaluated_candidate_count':fit.get('evaluated_candidate_count'),'charge':l.meter.compute_spent-before,'selected_schema':fit.get('selected_schema'),'train_rmse':fit.get('train_rmse'),'val_rmse':fit.get('val_rmse'),'model_sha256':fit.get('model_sha256')})
    return {'target':name, 'full_tag_count':len(full_tags), 'fits':out}


def main():
    suite = make_m3_1_factorized_walk_suite(SEED)
    base = M3_1FactorizedResearcher(gate=make_gate(), retrieval_mode='hipporag_ppr', damping_alpha=.65)
    bridge_calls = []
    orig_synth = m3.run_m3_workspace_synthesizer
    def traced_synth(workspace,lab,**kwargs):
        ans = orig_synth(workspace,lab,**kwargs)
        if lab.task_id == 'm3_bridge_fit':
            bridge_calls.append({'workspace':str(workspace),'costs':lab.meter.snapshot()})
        return ans
    m3.run_m3_workspace_synthesizer = traced_synth
    try:
        source_logs = base.form_and_revise_episodes('audit_eval01',suite,ROOT/'source_workspaces')
    finally:
        m3.run_m3_workspace_synthesizer = orig_synth
    (ROOT/'source_logs.json').write_text(json.dumps(source_logs,indent=2))
    (ROOT/'forest.json').write_text(json.dumps({k:getattr(base.forest,k) for k in ('nodes','local_implication_edges','horizontal_wormhole_edges')},indent=2))

    arms = {}
    for lbl, mode, a in [('Walk_Pool','hipporag_ppr',.65),('Alpha0_Pool','hipporag_alpha0_pool',0.),('Walk_NoPool','hipporag_walk_nopool',.65),('Alpha0_NoPool','hipporag_walk_nopool',0.)]:
        h = base.clone_with_mode(mode)
        h.damping_alpha=a
        h.build_or_refresh_index()
        h.gate=make_gate()
        arms[lbl]=h
    records=[]
    auth_capture=[]
    orig_auth = m3.authorize_abstraction_use
    current = {}
    def capture_auth(**kwargs):
        ans = orig_auth(**kwargs)
        if current.get('category')=='T_WalkBridge' and kwargs['requested_tier']=='local_solver_guidance':
            auth_capture.append({**current,'claim':kwargs['claim_contract'].get('conclusion_or_action',{}),'target_features':deepcopy(kwargs['target_features']),'authorization':ans})
        return ans
    m3.authorize_abstraction_use=capture_auth
    try:
        focus=[t for t in suite['targets'] if t[0] in {'T_Assoc','T_WalkBridge'}]
        for cat,name,(fac,ev) in focus:
            for label,agent in arms.items():
                current={'category':cat,'target':name,'arm':label}
                lab=fac(); ws=ROOT/'factor_targets'/label/name
                sub=agent.solve_target(name,lab,ws)
                evaluated=ev(ws,sub,lab.observations)
                records.append({**current,'evaluation':evaluated,'costs':lab.meter.snapshot(),'retrieval':deepcopy(agent.retrieval_logs[-1]),'submission':sub})
    finally:
        m3.authorize_abstraction_use=orig_auth

    # Recover the full empirically extracted child signatures which witness_only_signatures replaced.
    full_child_cues = {}
    full_cue_forest = deepcopy(base.forest)
    for ev in base.revision_events:
        nm=ev['witness_episode']; nid=ev['child_node_id']
        _,_,(fac,_) = next(x for x in suite['near_misses'] if x[0]==nm)
        lab=fac()
        ws=ROOT/'source_workspaces'/f'M3_nm_{nm}'
        lab.observations=[Observation(features=r['features'],target=r['target'],history=tuple(r.get('history',()))) for r in json.loads((ws/'observations.json').read_text())]
        lens=m3._estimate_observation_lens(lab,ROOT/'signature_recovery'/nm)
        sigs,_,_=extract_m3_1_observable_signatures(lab,lens)
        full_child_cues[nid]={'schema':tag(base.forest.nodes[nid]),'indexed':base.forest.nodes[nid]['probe_signatures'],'full_empirical':sigs}
        full_cue_forest.nodes[nid]['probe_signatures']=sigs

    cue_diagnostics=[]
    for _,name,(fac,_) in suite['walk_targets']:
        lab=fac(); ws=ROOT/'query_only'/name
        lens=m3._estimate_observation_lens(lab,ws)
        lab.acquire(build_m3_1_pilot_points(lab),workspace=ws,phase='presolve')
        q,_,_=extract_m3_1_observable_signatures(lab,lens)
        tv={**lab.task_view,'active_arity':len(m3._active_and_fixed_features(lab)[0])}
        variants={label:agent.index for label,agent in arms.items()}
        variants['Alpha0_Pool_full_empirical_child_cues']=HippoRAGWormholeIndex(full_cue_forest,damping_alpha=0.,wormhole_equivalence_pooling=True)
        variants['Flat_BM25']=FlatEqualMetadataIndex(base.forest)
        cue_diagnostics.append({'target':name,'query':q,'child_overlap':{nid:{'indexed':sorted(set(v['indexed'])&set(q)),'full_empirical':sorted(set(v['full_empirical'])&set(q))} for nid,v in full_child_cues.items()},'rankings':{label:[{'node_id':c['node_id'],'schema':tag(c['target_node']),'score':c['score'],'direct_signatures':c['matched_signatures'],'only_walk':c['reached_only_by_graph_walk']} for c in idx.retrieve(q,task_view=tv,top_k=6)['candidates']] for label,idx in variants.items()}})

    summ={}
    for label in arms:
        rr=[r for r in records if r['arm']==label]
        w=[r for r in rr if r['category']=='T_WalkBridge']
        summ[label]={'n':len(rr),'strict_pass':sum(r['evaluation']['strict_pass'] for r in rr),'recall':sum(r['retrieval'].get('confirmed_via_recall',False) for r in rr),'walk_recall':sum(r['retrieval'].get('confirmed_via_recall',False) for r in w),'measurement':mean(r['costs']['measurement_cost'] for r in rr),'compute':mean(r['costs']['compute_cost'] for r in rr),'composite':mean(r['costs']['composite_cost'] for r in rr),'walk_measurement':mean(r['costs']['measurement_cost'] for r in w),'walk_composite':mean(r['costs']['composite_cost'] for r in w)}
    result={'commit':COMMIT,'seed':SEED,'scope':'10 fixed T_Assoc/T_WalkBridge targets, one published evaluation seed; exact published DEV gate; post-hoc missing-factor diagnostic only.','factor_summary':summ,'records':records,'safe_high_load_authorizations':auth_capture,'source_bridge_calls':bridge_calls,'source_phases':{p:sum(r['phase']==p for r in source_logs) for p in sorted({r['phase'] for r in source_logs})},'indexed_vs_empirical_child_cues':full_child_cues,'query_diagnostics':cue_diagnostics,'k_input_invariance':check_k_inputs(),'pricing_control':pricing_control()}
    (ROOT/'factorization_control_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'factor_summary':summ,'source_bridge_calls':bridge_calls,'source_phases':result['source_phases'],'K_inputs_same':result['k_input_invariance']['all_target_inputs_identical'],'pricing':result['pricing_control']},indent=2))

if __name__=='__main__':
    main()
