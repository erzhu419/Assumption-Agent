"""Offline read-only controls for M6 b1d0c3e99. No model or Docker calls."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = (Path(os.environ.get('HEGEL_M6_REPO', str(Path(__file__).resolve().parents[2] / 'repo_m6_b1d0c3e99'))) / 'Hegel Machine')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from hegel_machine.cognition import m6_1_zero_leakage_ast_causal_benchmark as a
from hegel_machine.cognition import m6_frontier_real_benchmarks as f
from hegel_machine.cognition.m5_live_gpt_and_provenance import _execute_authored_python_in_isolated_subprocess, verify_decision_provenance_chain
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex

def sha(x): return hashlib.sha256(x.encode()).hexdigest()

def main():
    result = {}
    ev = ROOT/'docs/evidence/m6_1_zero_leakage_ast_causal_192ep_20261121'
    receipts = [json.loads(p.read_text()) for p in sorted(ev.rglob('episode_receipt.json'))]
    result['receipt_count'] = len(receipts)
    # Confirm hashes and chains; do not mistake them for semantic assurance.
    checks = {}
    for dirname in ['m6_1_zero_leakage_ast_causal_192ep_20261121','m6_live_gpt_causal_aufhebung_288ep_20261121','m6_phase0_causal_wormhole_sandbox_20261121']:
        e = ROOT/'docs/evidence'/dirname
        mismatches=[]; n=0
        for line in (e/'SHA256SUMS.txt').read_text().splitlines():
            if not line.strip(): continue
            h,p = line.split('  ',1); n+=1
            if hashlib.sha256((e/p).read_bytes()).hexdigest()!=h: mismatches.append(p)
        dags=list(e.rglob('decision_provenance_dag.json'))
        bad=[]
        for p in dags:
            dobj=json.loads(p.read_text())
            if isinstance(dobj,dict) and 'records' not in dobj and 'decision_dag' not in dobj:
                bad.append({'path':str(p.relative_to(e)),'schema_keys':list(dobj)})
                continue
            v=verify_decision_provenance_chain(dobj.get('records',dobj.get('decision_dag')) if isinstance(dobj,dict) else dobj)
            if not v['valid']: bad.append(str(p.relative_to(e)))
        checks[dirname]={'hash_files':n,'hash_mismatches':mismatches,'dags':len(dags),'invalid_dags':bad}
    result['integrity']=checks
    result['recomputed_aggregate']={}
    for model in sorted({r['model'] for r in receipts}):
        result['recomputed_aggregate'][model]={}
        for arm in sorted({r['arm'] for r in receipts}):
            rs=[r for r in receipts if r['model']==model and r['arm']==arm]
            ins=[r for r in rs if not r['is_out_of_scope']]
            result['recomputed_aggregate'][model][arm]={'in_scope_pass':sum(r['strict_pass'] for r in ins),'in_scope_n':len(ins),'false_confirmations':sum(r['false_confirmation'] for r in rs),'cost':sum(r['composite_cost'] for r in rs),'total_tokens':sum(r['total_tokens'] for r in rs),'failed_tasks':[r['task_id'] for r in ins if not r['strict_pass']]}
    # Re-run repository's seven pure tests without installing pytest.
    test_results=[]
    with tempfile.TemporaryDirectory(prefix='m6_focused_tests_') as td:
        for fn in ['test_m6_phase0_causal_wormhole_sandbox.py','test_m6_live_gpt_causal_aufhebung.py','test_m6_1_zero_leakage_and_bottleneck_fixes.py']:
            spec=importlib.util.spec_from_file_location(fn[:-3],ROOT/'tests'/fn)
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            for name,obj in vars(mod).items():
                if name.startswith('test_') and callable(obj):
                    try:
                        obj(Path(td)) if obj.__code__.co_argcount else obj()
                        test_results.append({'test':name,'passed':True})
                    except Exception as ex:
                        test_results.append({'test':name,'passed':False,'error':repr(ex)})
    result['repository_pure_tests']=test_results
    # AST collector does not follow helpers once predict contains any direct key.
    helper_code="def hidden(row):\n    return row['z_2']\ndef fit(rows):\n    pass\ndef predict(row):\n    return hidden(row) - .3*row['u_ctrl']\n"
    result['helper_dependency_omitted']=a.extract_and_translate_authored_ast(helper_code,{'u_ctrl':'u_ctrl'})
    nonrow_code="P={'z_2': 4.0}\ndef fit(rows):\n    pass\ndef predict(row):\n    return P['z_2'] * row['u_ctrl']\n"
    result['nonrow_dict_key_rewritten']=a.extract_and_translate_authored_ast(nonrow_code,{'z_2':'z_4','u_ctrl':'u_ctrl'})
    # Actual logged source assignments and completion-order dependence.
    w1=[r for r in receipts if r['wave']=='wave_1' and r['model']=='gpt-6.2-sol' and r['arm']=='Hegel_M6_1_ZeroLeakage_LiveGPT']
    result['actual_success_writeback_owners']=[{'task':r['task_id'],'owner':r['primary_recalled_node_id']} for r in receipts if r['wave']=='wave_1' and r['strict_pass'] and not r['is_out_of_scope'] and r['arm'].startswith('Hegel')]
    order_results={}
    for title,key in [('c01_first',lambda r:(0 if '__C01_' in r['task_id'] else 1,r['task_id'])),('c05_first',lambda r:(0 if '__C05_' in r['task_id'] else 1,r['task_id']))]:
        forest,archive=a.build_m6_1_clean_k40_forest()
        wb=a.consolidate_wave1_via_ast_wormholes(wave1_receipts=sorted(w1,key=key),arm='Hegel_M6_1_ZeroLeakage_LiveGPT',forest=forest,node_archive=archive)
        order_results[title]={'writeback':wb,'target_code':archive['node_C01_TreeF_doubleprime_catalysis'].get('confirmed_python_reference'),'target_active_vars':archive['node_C01_TreeF_doubleprime_catalysis'].get('confirmed_active_variables')}
    waves=build_m6_causal_24_waves()
    target=next(d for d in waves['wave_2'] if '__C01_' in d.task_id and 'Fdouble' in d.task_id)
    with tempfile.TemporaryDirectory(prefix='m6_order_control_') as td:
        for title,obj in order_results.items():
            evr=_execute_authored_python_in_isolated_subprocess(Path(td)/title,step_idx=1,python_code=obj['target_code'],train_rows=target.pilot_rows,confirm_rows=target.confirm_rows,heldout_rows=target.heldout_rows)
            obj['same_target_fit']={k:v for k,v in evr.items() if k not in ('model_path','exec_wall_ms')}
    result['completion_order_control']=order_results
    # An authored model can import the private generator and return evaluation answers.
    # The supplied train_rows are deliberately unused. This proves a boundary failure,
    # not an assertion that retained episodes exploited it.
    oracle="""from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves
_ds=next(d for d in build_m6_causal_24_waves()['wave_2'] if d.task_id==%r)
def _key(r): return tuple(sorted((k,v) for k,v in r.items() if k!='y'))
_truth={_key(r):r['y'] for r in _ds.pilot_rows+_ds.confirm_rows+_ds.heldout_rows}
def fit(rows): pass
def predict(row): return _truth[_key(row)]
""" % target.task_id
    with tempfile.TemporaryDirectory(prefix='m6_isolation_control_') as td:
        er=_execute_authored_python_in_isolated_subprocess(Path(td),step_idx=1,python_code=oracle,train_rows=target.pilot_rows,confirm_rows=target.confirm_rows,heldout_rows=target.heldout_rows)
        result['private_generator_import_control']={k:v for k,v in er.items() if k not in ('model_path','exec_wall_ms')}
    # Generic collinearity warning incorrectly treats selected observational probes as interventions.
    c03=next(d for d in waves['wave_2'] if '__C03_' in d.task_id)
    result['selected_only_probe_warning']=a.detect_observational_identifiability_hazards(c03.pilot_rows,c03.probe_pool)
    # Constant global DAG checks do not actually screen individual frontier lessons.
    forest,archive,meta=f.build_m6_frontier_general_forest()
    rec={'short_name':'test_shell_failure','wormhole_family':'fam_symbolic_and_operator_dynamics','strict_pass':False,'failure_taxonomy':'EXAMPLE','nogoods_recorded':[{'nogood_rung':'RUNG_1_SHELL_OR_ENV_QUIRK','assumption_refuted':'python alias exists','sublated_pivot':'use python3'}]}
    cons=f.consolidate_wave1_into_m6_forest(wave1_m6_summaries=[rec],forest=forest,archive=archive,meta_dag=meta)
    result['rung1_passes_causal_screen']=cons
    # Mock only external I/O, exercise the unchanged real loop and precommit gates.
    # A successful unrelated command falsely clears the verification debt.
    decision_queue=[{'action_type':'execute_in_remote_docker','bash_command':'write_bad_artifact'}, {'action_type':'execute_in_remote_docker','bash_command':'true'}, {'action_type':'submit_completed_task','final_answer':'done'}]
    class FakeClient:
        def __init__(self,**kwargs):pass
        def call_structured(self,**kwargs):
            p=decision_queue.pop(0);p.update({'rationale':'offline control','considered_alternatives':[],'root_contract_checklist':[],'nested_hypothesis_chain':[],'nogood_update':{'has_new_nogood':False}})
            return {'ok':True,'parsed':p,'wall_ms':0,'returned_model':'offline','prompt_tokens':0,'cached_tokens':0,'completion_tokens':0}
    snap_seq=[{}, {'/app/out.json':{'size':1,'sha16':'bad'}}, {'/app/out.json':{'size':1,'sha16':'bad'}}]
    task=f.FrontierBenchmarkTask(task_id='tb__offline_gate',short_name='offline_gate',benchmark='Terminal-Bench-Science-0.1',wave='wave_1',scientific_domain='audit',wormhole_family='audit',instruction_text='write a valid /app/out.json',agent_image_tag='unused',verifier_image_tag='unused',declared_artifacts=('/app/out.json',))
    forest,archive,meta=f.build_m6_frontier_general_forest()
    with tempfile.TemporaryDirectory(prefix='m6_gate_control_') as td, patch.object(f,'FrontierCodexClient',FakeClient), patch.object(f,'start_detached_container',return_value='offline'), patch.object(f,'stop_container_safely'), patch.object(f,'snapshot_container_files',side_effect=lambda cid:snap_seq.pop(0)), patch.object(f,'exec_container_bash',return_value={'stdout':'','stderr':'','exit_code':0,'timed_out':False,'wall_ms':0}), patch.object(f,'check_declared_artifacts_in_container',return_value={'artifact_status':{'/app/out.json':'FILE:1'},'all_present':True,'missing_or_empty':[]}), patch.object(f,'save_submitted_artifacts_snapshot',return_value=[]), patch.object(f,'run_tb_separate_verifier',return_value={'strict_pass':False,'reward':0,'artifact_audit':{'all_present':True},'ctrf_summary':{'failed_tests':[{'name':'bad_json','message':'invalid'}]},'verifier_timed_out':False}):
        sr=f.run_frontier_episode(task,'Hegel_M6_Frontier_LiveGPT',model='offline',codex_info={'provider_backend':'offline','auth_mode':'offline'},forest=forest,forest_archive=archive,hippo_idx=HippoRAGWormholeIndex(forest),episode_out_dir=Path(td),max_turns=14)
        result['unrelated_true_clears_verify_debt']={k:sr[k] for k in ('turns_used','ended_by_finish','pullback_gate_triggered','strict_pass')}
    (OUT/'controls_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'result_path':str(OUT/'controls_results.json'),'integrity':result['integrity'],'tests':test_results,'source_order':{k: {'vars':v['target_active_vars'],'fit':v['same_target_fit']} for k,v in order_results.items()},'private_generator':result['private_generator_import_control'],'unrelated_true':result['unrelated_true_clears_verify_debt']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
