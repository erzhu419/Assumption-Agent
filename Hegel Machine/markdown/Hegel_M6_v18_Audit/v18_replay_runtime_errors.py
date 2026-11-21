"""Reproduce the 60 saved runtime-error submissions in v18.4 failing episodes.

No model calls; no repository writes. Uses the saved training rows and y-free
prediction inputs. Only simple numeric imports are allowed for this diagnostic.
"""
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import ast, hashlib, json, os, subprocess, sys

ROOT=(Path(os.environ.get('HEGEL_V18_REPO', str(Path(__file__).resolve().parent.parent / 'repo_v18_922b95e19'))) / 'Hegel Machine')
OUT=Path(__file__).resolve().parent
if len(sys.argv)>1 and sys.argv[1]=='--worker':
    import traceback
    code_path=Path(sys.argv[2]); payload=json.loads(Path(sys.argv[3]).read_text())
    try:
        ns={};exec(compile(code_path.read_text(),str(code_path),'exec'),ns)
        state=ns['fit'](payload['train_rows'])
        for kind,rows in [('train',payload['train_rows']),('confirm',payload['confirm_inputs']),('heldout',payload['heldout_inputs'])]:
            for row in rows:float(ns['predict']({k:v for k,v in row.items() if k!='y'}))
        print(json.dumps({'ok':True,'note':'No runtime error in this environment'}))
    except BaseException as e:
        print(json.dumps({'ok':False,'exception_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}))
    raise SystemExit(0)

jobs=[]
for dp in sorted((ROOT/'docs/evidence/v18_4_hegel_discovery_240_20261121').rglob('decision_provenance_dag.json')):
    d=json.loads(dp.read_text()); r=d['episode_summary']
    if r['strict_pass']:continue
    for rec in d['decision_dag']:
        if rec['decision_type']!='D7_PYTHON_SUBPROCESS_EXECUTION_ERROR':continue
        model_name=Path(rec['extra_artifacts']['model_path']).name
        step=model_name.split('_')[-1].split('.')[0]
        codep=dp.parent/model_name; payload=dp.parent/f'eval_payload_{step}.json'
        jobs.append({'model':r['model'],'arm':r['arm'],'task_id':r['task_id'],'code_path':str(codep),'payload_path':str(payload),
            'original_model_sha256':rec['evidence_inputs']['model_sha256'],
            'original_recorded_error':rec['rejection_reasons']})

def replay(j):
    try:
        tree=ast.parse(Path(j['code_path']).read_text())
    except SyntaxError as e:
        return {**j,'ok':False,'exception_type':'SyntaxError','error':str(e)}
    imps=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imps.extend(a.name.split('.')[0] for a in node.names)
        elif isinstance(node,ast.ImportFrom):imps.append((node.module or '').split('.')[0])
    if set(imps)-{'math','numpy','scipy','typing','functools','itertools','json'}:
        return {**j,'skipped_imports':imps}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    try:
        p=subprocess.run([sys.executable,__file__,'--worker',j['code_path'],j['payload_path']],capture_output=True,text=True,timeout=15,env=env,cwd=OUT)
        lines=[x for x in p.stdout.splitlines() if x.strip()]
        res=json.loads(lines[-1]) if lines else {'returncode':p.returncode,'stderr':p.stderr[-1200:]}
        return {**j,**res}
    except subprocess.TimeoutExpired:return {**j,'timeout_seconds':15}
checkpoint_path=OUT/'v18_runtime_error_replay_checkpoint.jsonl'
done={r['code_path']:r for r in (json.loads(x) for x in checkpoint_path.read_text().splitlines())} if checkpoint_path.exists() else {}
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(replay,j) for j in jobs if j['code_path'] not in done]
    for future in as_completed(futures):
        r=future.result();done[r['code_path']]=r
        with checkpoint_path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        if len(done)%10==0:print('saved',len(done),'/',len(jobs),flush=True)
results=[]
for j in jobs:
    r={**done[j['code_path']],**j}
    r['replayed_model_sha256']=hashlib.sha256(Path(j['code_path']).read_bytes()).hexdigest()
    r['model_hash_matches_original']=r['replayed_model_sha256']==r['original_model_sha256']
    results.append(r)
summary=Counter((r.get('exception_type') or ('ok' if r.get('ok') else 'other'))+': '+str(r.get('error','')) for r in results)
(OUT/'v18_replayed_runtime_errors.json').write_text(json.dumps({'count':len(results),'summary':dict(summary),
    'environment':{'python':sys.version,'timeout_seconds':15,'original_host_timeout_seconds':25,
                   'note':'Local retrospective contract replay; environment and timeout differ from original. No leaderboard results replaced.'},
    'all_model_hashes_match_original':all(r['model_hash_matches_original'] for r in results),
    'results':results},ensure_ascii=False,indent=2))
print(json.dumps({'count':len(results),'summary':dict(summary)},ensure_ascii=False,indent=2))
