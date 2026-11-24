"""Offline, read-only evidence recomputation; never invokes model or Docker."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, os

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=os.environ.get('HEGEL_AUDIT_REPO'),help='Checkout of Assumption-Agent at 4001bef56; also accepts HEGEL_AUDIT_REPO.')
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=(args.repo_root or Path(__file__).resolve().parents[2]/'repo_m691_4001bef56').resolve()
EV=ROOT/'Hegel Machine/docs/evidence'
if not EV.is_dir():parser.error('Repository not found: pass --repo-root /path/to/Assumption-Agent or HEGEL_AUDIT_REPO.')
def verify_decision_provenance_chain(records):
    """Independent canonical-JSON chain/parent/token/cost verification."""
    prev='0'*64; ids=set(); sums={}
    for seq,r in enumerate(records,1):
        if r.get('decision_seq')!=seq or r.get('prev_record_sha256')!=prev:return {'valid':False}
        sha=hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='record_sha256'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if sha!=r.get('record_sha256'):return {'valid':False}
        parent=r.get('parent_decision_id')
        if (seq==1 and parent is not None) or (seq>1 and parent not in ids):return {'valid':False}
        for delta,total in [('measurement_delta','measurement_cost'),('compute_delta','compute_cost'),('prompt_tokens_delta','prompt_tokens'),('cached_tokens_delta','cached_tokens'),('completion_tokens_delta','completion_tokens')]:
            sums[total]=round(sums.get(total,0)+r['cost_ledger_delta'].get(delta,0),6)
            if abs(sums[total]-r['cumulative_cost_after'].get(total,0))>1e-6:return {'valid':False}
        ids.add(r['decision_id']);prev=sha
    return {'valid':bool(records)}

BUNDLES=[EV/x for x in ['m6_8_b_prime_frontier_40ep_20261122','m6_8_1_b_prime_frontier_40ep_20261122','m6_9_a_two_branch_frontier_40ep_20261122','m6_9_1_a_two_branch_frontier_40ep_20261123']]
rows=[]; checks=[]; trace_index=[]
def read(p): return json.loads(p.read_text())
for b in BUNDLES:
    ck={'bundle':b.name,'sha_count':0,'sha_mismatches':[],'dag_count':0,'bad_chains':[],'bad_summary_tokens':[],'summary_verifier_mismatches':[], 'activation_mismatches':[]}
    for line in (b/'SHA256SUMS.txt').read_text().splitlines():
        sha,rel=line.split('  ',1);p=b/rel;ck['sha_count']+=1
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:ck['sha_mismatches'].append(rel)
    for p in sorted(b.rglob('00_episode_summary.json')):
        ep=p.parent; s=read(p); dg=read(ep/'01_decision_provenance_dag.json'); rs=dg['records']; ck['dag_count']+=1
        post=ep/'03_postlock_branch_verifier_report.json'; orig=read(ep/'03_official_verifier_report.json'); v=read(post) if post.exists() else orig
        scope='branch' if 'branch_trajectories' in str(p) else 'top'
        label='Thesis' if ep.name=='branch_1_thesis_bare' else 'Antithesis' if ep.name=='branch_2_antithesis_wormhole' else 'Bare_R1_copy' if s['arm']=='Bare_ReAct_LiveGPT' else 'Synthesis' if 'TwoBranch' in s['arm'] else 'Hegel'
        if not verify_decision_provenance_chain(rs)['valid']:ck['bad_chains'].append(str(ep.relative_to(b)))
        if s['strict_pass']!=orig['strict_pass']:ck['summary_verifier_mismatches'].append(str(ep.relative_to(b)))
        for name in ('prompt_tokens','completion_tokens','cached_tokens'):
            n=sum(x.get('cost_ledger_delta',{}).get(name+'_delta',0) for x in rs)
            if n!=s[name]:ck['bad_summary_tokens'].append([str(ep.relative_to(b)),name,n,s[name]])
        full=ep/'04c_full_skill_activations.jsonl'
        calls=[json.loads(l) for l in full.read_text().splitlines() if l.strip()] if full.exists() else s.get('skill_activations',[])
        if sum(c.get('call_count',1) for c in calls)!=s.get('skill_activation_count',0):ck['activation_mismatches'].append([str(ep.relative_to(b)),sum(c.get('call_count',1) for c in calls),s.get('skill_activation_count',0)])
        rr={k:s.get(k) for k in ('task_id','wave','arm','turns_used','max_turns','total_tokens','prompt_tokens','cached_tokens','completion_tokens','effective_billed_tokens','wall_ms','skill_activation_count','nogood_count','host_wormhole_gate_executed','incumbent_auto_restored','final_candidate_sha256','validated_incumbent_sha256','terminal_record_sha256','failure_taxonomy','producer_skill_manifest_summary','staged_wormhole_modules')}
        rr.update(bundle=b.name,scope=scope,label=label,native_pass=bool(v['strict_pass']),original_deferred=orig.get('failure_taxonomy')=='DEFERRED_FOR_BLIND_SYNTHESIS',source=str(ep.relative_to(ROOT)),dag_sha256=hashlib.sha256((ep/'01_decision_provenance_dag.json').read_bytes()).hexdigest(),verifier_source=str((post if post.exists() else ep/'03_official_verifier_report.json').relative_to(ROOT)),verifier_wall_ms=v.get('verifier_wall_ms',0),failed_tests=v.get('ctrf_summary',{}).get('failed_tests',[]),verifier_taxonomy=v.get('failure_taxonomy'),selection=s.get('synthesis_decision'),branch_outcomes=s.get('postlock_branch_outcomes'),judge_total_tokens=v.get('llm_judge_tokens',{}).get('total_tokens',0),regrade_receipts=[x.name for x in ep.glob('*regrade*')],calls_by_function=dict(Counter({k:sum(z.get('call_count',1) for z in calls if z.get('function')==k) for k in {z.get('function') for z in calls}})),calls_by_module=dict(Counter({k:sum(z.get('call_count',1) for z in calls if z.get('module')==k) for k in {z.get('module') for z in calls}})),calls_by_role=dict(Counter({k:sum(z.get('call_count',1) for z in calls if z.get('caller_role')==k) for k in {z.get('caller_role') for z in calls}})),calls_by_status=dict(Counter({k:sum(z.get('call_count',1) for z in calls if z.get('status')==k) for k in {z.get('status') for z in calls}})),synthesis_tokens=sum(x.get('cost_ledger_delta',{}).get('prompt_tokens_delta',0)+x.get('cost_ledger_delta',{}).get('completion_tokens_delta',0) for x in rs if x['decision_type']=='D3_DIALECTICAL_SYNTHESIS_COMMIT'),synthesis_wall_ms=sum(x.get('cost_ledger_delta',{}).get('wall_ms_delta',0) for x in rs if x['decision_type']=='D3_DIALECTICAL_SYNTHESIS_COMMIT'))
        rows.append(rr)
        trace_index.append({'source':rr['source'],'task':rr['task_id'],'label':label,'native_pass':rr['native_pass'],'records':[{'seq':x['decision_seq'],'type':x['decision_type'],'step':x.get('step_idx'),'inputs':x.get('evidence_inputs'),'chosen':x.get('chosen_option'),'rationale':x.get('rationale_summary'),'extra':x.get('extra_artifacts')} for x in rs]})
    checks.append(ck)
groups=defaultdict(list)
for r in rows:groups[r['dag_sha256']].append(r['source'])
agg=[]
for bundle,label in sorted({(r['bundle'],r['label']) for r in rows}):
    rr=[r for r in rows if (r['bundle'],r['label'])==(bundle,label)]
    agg.append(dict(bundle=bundle,label=label,n=len(rr),native_pass=sum(r['native_pass'] for r in rr),wave1=sum(r['native_pass'] and r['wave']=='wave_1' for r in rr),wave2=sum(r['native_pass'] and r['wave']=='wave_2' for r in rr),tb=sum(r['native_pass'] and r['task_id'].startswith('tb') for r in rr),**{k:sum(r.get(k,0) or 0 for r in rr) for k in ('turns_used','total_tokens','prompt_tokens','cached_tokens','completion_tokens','effective_billed_tokens','wall_ms','verifier_wall_ms','skill_activation_count','synthesis_tokens','synthesis_wall_ms','judge_total_tokens')}))
latest=BUNDLES[-1].name;old=BUNDLES[-2].name
by_key={(r['bundle'],r['label'],r['task_id']):r for r in rows}
mat=[]
for task in sorted({r['task_id'] for r in rows}):
    m={'task_id':task}
    for b,lab,key in [(latest,'Bare_R1_copy','Bare_R1'),(old,'Thesis','Bare_R2'),(latest,'Thesis','Bare_R3'),(BUNDLES[0].name,'Hegel','M68'),(BUNDLES[1].name,'Hegel','M681'),(old,'Antithesis','M69_anti'),(old,'Synthesis','M69_synth'),(latest,'Antithesis','M691_anti'),(latest,'Synthesis','M691_synth')]:
        r=by_key[(b,lab,task)];m[key]=int(r['native_pass'])
    m['Bare_oracle_R1_R3']=max(m['Bare_R1'],m['Bare_R3']);m['Bare_oracle_R1_R2_R3']=max(m['Bare_R1'],m['Bare_R2'],m['Bare_R3']);m['M691_branch_union']=max(m['Bare_R3'],m['M691_anti']); m['M691_mechanism']=by_key[(latest,'Synthesis',task)]['selection']['selection_mechanism'];mat.append(m)
result={'checks':checks,'aggregates':agg,'episode_occurrences':len(rows),'unique_dag_bytes':len(groups),'duplicate_dags':{k:v for k,v in groups.items() if len(v)>1},'per_task':mat,'latest_union_regret':sum(m['M691_branch_union']-m['M691_synth'] for m in mat)}
for name,obj in [('recomputed_results.json',result),('episode_index.json',rows),('trace_index.json',trace_index)]: (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
with (OUT/'per_task_matrix.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(mat[0]));w.writeheader();w.writerows(mat)
with (OUT/'episode_matrix.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['bundle','label','task_id','wave','native_pass','turns_used','total_tokens','effective_billed_tokens','wall_ms','skill_activation_count','source'],extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in result.items() if k not in ('duplicate_dags','per_task')},ensure_ascii=False,indent=2))
