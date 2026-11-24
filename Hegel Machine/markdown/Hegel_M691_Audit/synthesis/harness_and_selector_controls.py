"""Offline controls for commit 4001bef56; no Docker, model, or network calls.

The exact embedded validator and symbolic evaluator are executed against disposable
local fixtures. Docker invocation alone is redirected to the local Python process.
The symbolic fixture path is substituted for /app; the evaluator logic is unchanged.
"""
from __future__ import annotations
import ast, importlib.util, inspect, json, os, sys, tempfile, traceback
from pathlib import Path
from unittest.mock import patch
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HEGEL_AUDIT_REPO', os.environ.get('HEGEL_M691_REPO', str(HERE.parents[1] / 'repo_m691_4001bef56'))))
HM = REPO / 'Hegel Machine'
sys.path.insert(0, str(HM / 'src'))
from hegel_machine.cognition import m6_frontier_real_benchmarks as f
from hegel_machine.cognition import m6_8_b_prime_capability_bridge as b
from hegel_machine.cognition import m6_9_a_dialectical_synthesis_engine as s

result = {'commit': '4001bef56ffa77520c248b853e7fed3535ee9fe4', 'scope': 'offline concrete controls; historical scores not modified'}
commands = [
 'true', 'python3 -c "print(\'solver.py\')"', 'pytest -q /tmp/test_unrelated.py',
 'python3 /tmp/validate_saved_candidate.py',
 'python3 -c "import math; print(\'solver.py\')"',
 'python3 -c "from pathlib import Path; Path(\'/app/solver.py\').write_text(\'raise RuntimeError(1)\')"',
 'pytest -q /tmp/test_unrelated_validator.py',
]
result['verification_command_acceptance'] = [{'command':x, 'accepted':f.is_substantive_verification_command(x,('/app/solver.py',))} for x in commands]

# Real validator body, zero Docker.
orig_run = subprocess.run
def local_validator(cmd, **kwargs):
 assert cmd[:4] == ['docker','exec','-w','/'], cmd
 assert cmd[5:7] == ['python3','-c'], cmd
 return orig_run([sys.executable,'-c',cmd[7],cmd[8]], **kwargs)
result['contract_controls'] = []
for label, filename, content in [
 ('syntax_valid_but_import_crashes','solver.py','raise RuntimeError("candidate is not executable")\n'),
 ('missing_dependency','solver.py','import this_module_does_not_exist_for_hegel_audit\ndef predict(x): return x\n'),
 ('valid_syntax_no_required_entrypoint','solver.py','answer = 42\n'),
 ('nan_csv','results.csv','value\nnan\n'),
 ('nan_json','results.json','{"value": NaN}\n'),
]:
 with tempfile.TemporaryDirectory(prefix='hegel_contract_') as td:
  p=Path(td)/filename;p.write_text(content)
  with patch.object(b.subprocess,'run',local_validator):
   receipt=b.verify_candidate_artifact_contract_in_container('offline',(str(p),),'not-the-real-digest')
  result['contract_controls'].append({'label':label, **receipt})

# Digest: the exact function still ends with a character slice.
with tempfile.TemporaryDirectory(prefix='hegel_digest_') as td:
 ep=Path(td)
 (ep/'02_human_readable_trajectory.md').write_text('# Head\n\n### Turn 1 — long diagnostic\n```python\n'+'X'*17000+'\n```\n\n### Turn 2 — submit\nDone\n## 4. Official Verifier\nSECRET\n')
 d=json.loads(s._extract_trajectory_evidence_digest(ep,{},max_chars=14000))
 tail=d['trajectory_tail_public_evidence']
 result['long_turn_digest']={'length':len(tail),'starts_at_turn_boundary':tail.startswith('### Turn '),'start_excerpt':tail[:70],'private_footer_absent':'SECRET' not in tail}

# Symbolic benchmark executes no prediction for the actual supported task ABI.
bench_tree=ast.parse(s.PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY)
sr_ast=next(n for n in ast.walk(bench_tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sr_code' for t in n.targets))
sr_code=ast.literal_eval(sr_ast.value)
result['symbolic_evaluator_abi'] = []
with tempfile.TemporaryDirectory(prefix='hegel_srabi_') as td:
 import numpy as np
 p=Path(td); y=np.tile([0,1],40); X=np.column_stack((y, np.arange(len(y))))
 np.savez(p/'train.npz',X=X,y=y)
 (p/'regressor.py').write_text('def predict(X_train,y_train,X_test):\n    raise AssertionError("actual task API called")\n')
 for mode, expr in [('preallocated_zero','np.zeros_like(y)'),('preallocated_truth','np.array(y,copy=True)')]:
  # Patch only allocation contents; a valid evaluator must overwrite every OOF row.
  code=sr_code.replace('oof = np.empty_like(y)', 'oof = '+expr).replace('/app',td)
  pr=orig_run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15)
  result['symbolic_evaluator_abi'].append({'allocation':mode,'exit_code':pr.returncode,'stdout':pr.stdout.strip(),'stderr':pr.stderr[-500:],'predict_was_called': 'actual task API called' in pr.stderr})

# Existing exported tests, directly invoking test functions because pytest is not installed.
class MinimalMonkeyPatch:
 def __init__(self):self.ps=[]
 def setattr(self,obj,name,value):
  p=patch.object(obj,name,value);p.start();self.ps.append(p)
 def undo(self):
  for p in reversed(self.ps):p.stop()
result['existing_test_functions_direct'] = []
for fname, selected in [
 ('test_m6_9_a_dialectical_synthesis.py', None),
 ('test_m6_8_b_prime_audit_closure.py', {'test_frontier_harness_commit_gate_reentrant_rwlock_and_m5_reconciliation'}),
]:
 spec=importlib.util.spec_from_file_location('audit_'+Path(fname).stem,HM/'tests'/fname)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for name,func in sorted(vars(mod).items()):
  if not name.startswith('test_') or not callable(func) or (selected is not None and name not in selected):continue
  mp=MinimalMonkeyPatch()
  try:
   kwargs={'monkeypatch':mp} if 'monkeypatch' in inspect.signature(func).parameters else {}
   func(**kwargs); outcome={'name':name,'passed':True}
  except Exception as exc:outcome={'name':name,'passed':False,'error':repr(exc),'traceback':traceback.format_exc()[-1800:]}
  finally:mp.undo()
  result['existing_test_functions_direct'].append(outcome)

# Run exact public harness gating paths against mocks (never Docker/model).
class FakeClient:
 responses=[]
 def __init__(self,**kwargs):self.remaining=iter(self.responses)
 def call_structured(self,**kwargs):
  try: action,command=next(self.remaining)
  except StopIteration: action,command='submit_completed_task',None
  return {'ok':True,'wall_ms':0.0,'prompt_tokens':0,'cached_tokens':0,'completion_tokens':0,'returned_model':'offline_mock','parsed':{'action_type':action,'bash_command':command,'final_answer':'done','rationale':'Offline control','considered_alternatives':[],'active_hypothesis_and_worst_slice_metric':'','nogood_update':{}}}
class FakeIndex:
 def retrieve(self,*args,**kwargs):return {'candidates':[],'query_compute_cost':0.0}
def noop(*a,**k):return {'stdout':'','stderr':'','exit_code':0,'timed_out':False,'wall_ms':0.0}
def verifier(*a,**k):return {'strict_pass':False,'reward':0.0,'verifier_wall_ms':0.0,'artifact_audit':{'all_present':True},'ctrf_summary':{},'verifier_stdout':'','verifier_stderr':'','verifier_timed_out':False}
task=f.FrontierBenchmarkTask(task_id='offline_audit',short_name='offline_audit',benchmark='Terminal-Bench-Science-0.1',wave='wave_1',scientific_domain='audit',wormhole_family='audit',instruction_text='Write a solver.',agent_image_tag='mock',verifier_image_tag='mock',declared_artifacts=('/app/solver.py',))
result['harness_submit_controls']=[]
for label,cap,steps in [
 ('repeat_submit',5,[('submit_completed_task',None)]*5),
 ('print_only',5,[('execute_in_remote_docker',commands[1]),('submit_completed_task',None)]),
 ('last_turn_submit',1,[('submit_completed_task',None)]),
 ('irrelevant_import_plus_print',5,[('execute_in_remote_docker',commands[4]),('submit_completed_task',None)]),
]:
 FakeClient.responses=steps;forest,archive,_=f.build_m6_frontier_general_forest()
 with tempfile.TemporaryDirectory(prefix='hegel_mock_ep_') as td:
  with patch.multiple(f,FrontierCodexClient=FakeClient,start_detached_container=lambda *a,**k:'mock',stop_container_safely=lambda *a,**k:None,exec_container_bash=noop,snapshot_container_files=lambda *a,**k:{},compute_declared_artifacts_digest=lambda *a,**k:'a'*64,check_declared_artifacts_in_container=lambda *a,**k:{'all_present':True,'missing_or_empty':[],'artifact_status':{'/app/solver.py':'FILE:40'}},find_uninlined_wormhole_imports_in_container=lambda *a,**k:[],harvest_container_skill_activations=lambda *a,**k:[],save_submitted_artifacts_snapshot=lambda *a,**k:[],run_tb_separate_verifier=verifier):
   d=f.run_frontier_episode(task,'Hegel_M6_9_Antithesis_LiveGPT',model='offline_mock',codex_info={'provider_backend':'offline_mock','auth_mode':'offline_mock'},forest=forest,forest_archive=archive,hippo_idx=FakeIndex(),episode_out_dir=Path(td),max_turns=cap)
  result['harness_submit_controls'].append({'case':label,'max_turns':cap,**{k:d[k] for k in ['ended_by_finish','turns_used','pullback_gate_triggered','strict_pass','provenance_verified']}})

(HERE/'harness_and_selector_controls.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
