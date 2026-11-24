"""Read-only recomputation of M6.3--M6.7 receipts. No model or Docker calls."""
from pathlib import Path
import collections,csv,hashlib,json,os,re,sys

ROOT = Path(os.environ.get('HEGEL_M67_REPO', str(Path(__file__).resolve().parents[2] / 'repo_m67_948e2d57a')))
if not ROOT.exists(): ROOT=Path(__file__).resolve().parents[3]/'repo_m67_948e2d57a'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'Hegel Machine/src'))
from hegel_machine.cognition.m5_live_gpt_and_provenance import verify_decision_provenance_chain
EV=ROOT/'Hegel Machine/docs/evidence'
BUNDLES=sorted(p for p in EV.glob('m6_*') if re.match(r'm6_[1-7](?:_|a_)',p.name) and ('frontier' in p.name or 'ablation' in p.name))
rows=[]; checks=[]; traces=[]
for bundle in BUNDLES:
    check={'bundle':bundle.name,'sha_count':0,'sha_mismatch':[], 'bad_chain':[], 'summary_verifier_mismatch':[], 'summary_ledger_mismatch':[], 'activation_mismatch':[]}
    sha=bundle/'SHA256SUMS.txt'
    if sha.exists():
        for line in sha.read_text().splitlines():
            digest,rel=line.split('  ',1);p=bundle/rel;check['sha_count']+=1
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:check['sha_mismatch'].append(rel)
    for p in sorted((bundle/'episodes').glob('*/00_episode_summary.json')):
        ep=p.parent;s=json.loads(p.read_text());v=json.loads((ep/'03_official_verifier_report.json').read_text());d=json.loads((ep/'01_decision_provenance_dag.json').read_text());rs=d['records']
        if not verify_decision_provenance_chain(rs)['valid']:check['bad_chain'].append(ep.name)
        if s['strict_pass']!=v['strict_pass']:check['summary_verifier_mismatch'].append(ep.name)
        for name in ['prompt_tokens','completion_tokens','cached_tokens']:
            n=sum(x.get('cost_ledger_delta',{}).get(name+'_delta',0) for x in rs)
            if n!=s[name]:check['summary_ledger_mismatch'].append([ep.name,name,n,s[name]])
        actual_calls=sum(x.get('call_count',1) for x in s.get('skill_activations',[]))
        if actual_calls!=s.get('skill_activation_count',0):check['activation_mismatch'].append([ep.name,actual_calls,s.get('skill_activation_count')])
        row={k:v for k,v in s.items() if k not in ['skill_activations','staged_wormhole_modules','nogoods_recorded']}
        row.update(bundle=bundle.name,source=str(ep.relative_to(ROOT)),dag_sha256=hashlib.sha256((ep/'01_decision_provenance_dag.json').read_bytes()).hexdigest(),verifier_pass=v['strict_pass'],verifier_wall_ms=v.get('verifier_wall_ms'),verifier_timed_out=v.get('verifier_timed_out'),ctfr=v.get('ctrf_summary',{}).get('summary',{}),failed_tests=v.get('ctrf_summary',{}).get('failed_tests',[]),gate_reports=s.get('host_wormhole_gate_reports',[]),activation_by_function=dict(collections.Counter({f:sum(z.get('call_count',1) for z in s.get('skill_activations',[]) if z.get('function')==f) for f in set(z.get('function') for z in s.get('skill_activations',[]))})))
        rows.append(row)
        if 'Hegel' in s['arm']:
            traces.append({'source':row['source'],'task':s['task_id'],'arm':s['arm'],'passed':s['strict_pass'],'records':[{'type':r['decision_type'],'step':r.get('step_idx'),'rationale':r.get('rationale_summary'),'chosen':r.get('chosen_option'),'extra':r.get('extra_artifacts',{}),'ledger':r.get('cost_ledger_delta',{})} for r in rs]})
    checks.append(check)
agg=[]
for bundle,arm in sorted(set((r['bundle'],r['arm']) for r in rows)):
    rr=[r for r in rows if (r['bundle'],r['arm'])==(bundle,arm)]
    agg.append({'bundle':bundle,'arm':arm,'n':len(rr),'summary_passes':sum(r['strict_pass'] for r in rr),'verifier_passes':sum(r['verifier_pass'] for r in rr),'w1':sum(r['strict_pass'] and r['wave']=='wave_1' for r in rr),'w2':sum(r['strict_pass'] and r['wave']=='wave_2' for r in rr),'turns':sum(r['turns_used'] for r in rr),'tokens':sum(r['total_tokens'] for r in rr),'effective_tokens':sum(r['effective_billed_tokens'] for r in rr),'wall_ms':sum(r['wall_ms'] for r in rr),'verifier_wall_ms':sum(r.get('verifier_wall_ms',0) or 0 for r in rr),'skills':sum(r.get('skill_activation_count',0) for r in rr),'skill_tasks':sum(r.get('skill_activation_count',0)>0 for r in rr),'real_gate_report_tasks':sum(bool(r.get('gate_reports')) for r in rr),'tb':sum(r['strict_pass'] and r['task_id'].startswith('tb') for r in rr)})
groups=collections.defaultdict(list)
for r in rows:groups[r['dag_sha256']].append(r['source'])
matrix=[]
for task in sorted(set(r['task_id'] for r in rows)):
    rr={'task_id':task}
    for r in rows:
        if r['task_id']==task:
            key='Bare' if r['arm']=='Bare_ReAct_LiveGPT' else re.match(r'(m6_\d[a-z]?)',r['bundle']).group(1)
            rr[key]={'pass':r['strict_pass'],'verifier_pass':r['verifier_pass'],'turns':r['turns_used'],'skills':r.get('skill_activation_count',0),'gate_reports':r.get('gate_reports'),'failure':r['failure_taxonomy'],'source':r['source']}
    matrix.append(rr)
result={'episodes':len(rows),'unique_dag_hashes':len(groups),'checks':checks,'aggregates':agg,'matrix':matrix,'duplicates':{k:v for k,v in groups.items() if len(v)>1},'union_hegel':sorted(set(r['task_id'] for r in rows if 'Hegel' in r['arm'] and r['strict_pass']))}
for name,obj in [('frontier_recomputed.json',result),('episode_index.json',rows),('trace_digest.json',traces)]:
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
with (OUT/'episode_matrix.csv').open('w') as f:
    fields=['bundle','arm','task_id','strict_pass','verifier_pass','turns_used','total_tokens','wall_ms','skill_activation_count','source'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({k:result[k] for k in ['episodes','unique_dag_hashes','checks','aggregates','union_hegel']},ensure_ascii=False,indent=2))
