"""Read-only recomputation of committed v18.1-v18.4 evidence; no model calls."""
from pathlib import Path
import os
from collections import defaultdict, Counter
import hashlib, json, csv, sys

ROOT=(Path(os.environ.get('HEGEL_V18_REPO', str(Path(__file__).resolve().parent.parent / 'repo_v18_922b95e19'))) / 'Hegel Machine')
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from hegel_machine.cognition.m5_live_gpt_and_provenance import verify_decision_provenance_chain

def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def summarize(rows):
    return dict(n=len(rows), in_scope=sum(not r['is_out_of_scope'] for r in rows),
        passed=sum(r['strict_pass'] and not r['is_out_of_scope'] for r in rows),
        oos=sum(r['is_out_of_scope'] and r['status']=='scoped_unknown' for r in rows),
        false=sum(r['false_confirmation'] for r in rows),
        M=round(sum(r['measurement_cost'] for r in rows),5),
        K=round(sum(r['compute_cost'] for r in rows),5),
        C=round(sum(r['composite_cost'] for r in rows),5),
        calls=sum(r['api_calls'] for r in rows),
        effective_tokens=sum(r['effective_billed_tokens'] for r in rows),
        wall_ms=round(sum(r['wall_ms'] for r in rows),2))

def main():
    result={'sha':{},'chains':{},'deterministic_chains':{},'groups':{},'failure_counts':{},'failure_cases':[], 'repricing':{}}
    for root in sorted((ROOT/'docs/evidence').glob('v18_[1234]*')):
        manifest=root/'SHA256SUMS.txt'
        missing=[];mismatch=[];num=0
        for line in manifest.read_text().splitlines():
            if not line.strip():continue
            expected,rel=line.split('  ',1);p=root/rel;num+=1
            if not p.exists():missing.append(rel)
            elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:mismatch.append(rel)
        result['sha'][root.name]={'entries':num,'missing':missing,'mismatch':mismatch}
        dags=list(root.rglob('*provenance_dag.json'));bad=[];nrecords=0
        for dag in dags:
            d=json.loads(dag.read_text());chain=d.get('decision_dag') or d.get('records') or []
            checked=verify_decision_provenance_chain(chain)
            nrecords+=len(chain)
            if not checked['valid']:bad.append({'path':str(dag.relative_to(root)),'result':checked})
        result['chains'][root.name]={'files':len(dags),'records':nrecords,'invalid':bad}
        for ledger_path in root.glob('deterministic*provenance_ledger.jsonl'):
            grouped=defaultdict(list)
            for rec in load_jsonl(ledger_path):grouped[(rec['run_id'],rec['model'],rec['arm'],rec['task_id'])].append(rec)
            bad=[]
            for key,chain in grouped.items():
                checked=verify_decision_provenance_chain(chain)
                if not checked['valid']:bad.append({'key':key,'result':checked})
            result['deterministic_chains'][root.name]={'episodes':len(grouped),'records':sum(map(len,grouped.values())),'invalid':bad}
    files={
        'v18.1':'v18_1_m5_live_gpt_and_decision_provenance_20261120/live_gpt_episode_receipts.jsonl',
        'v18.2':'v18_2_m5_sovereign_closure_and_live_subscription_20261121/live_challenge_L07_L08_receipts.jsonl',
        'v18.3_fresh':'v18_3_m5_complete_3model_continual_complex_and_amortization_20261121/stage1_post_audit_fresh_receipts.jsonl',
        'v18.3_CX':'v18_3_m5_complete_3model_continual_complex_and_amortization_20261121/stage2_continual_complex_receipts.jsonl',
        'v18.4':'v18_4_hegel_discovery_240_20261121/all_completed_episode_receipts.jsonl',
    }
    allrows=[]
    for label,rel in files.items():
        rows=load_jsonl(ROOT/'docs/evidence'/rel);groups=defaultdict(list)
        for r in rows:groups[(r['model'],r['arm'])].append(r)
        result['groups'][label]={'|'.join(k):summarize(v) for k,v in groups.items()}
        for r in rows:allrows.append({'version':label,**r})
    root=ROOT/'docs/evidence/v18_4_hegel_discovery_240_20261121'
    rows=[r for r in allrows if r['version']=='v18.4'];fam=defaultdict(list);wave=defaultdict(list)
    cases=[];pricing=defaultdict(lambda:Counter());patterns=Counter();oos_decisions=Counter();provider_errors=Counter();provider_by_arm=Counter()
    for r in rows:
        fam[(r['model'],r['arm'],r['subfamily'])].append(r)
        wave[(r['model'],r['arm'],r['continual_wave'])].append(r)
        ws=root/'workspaces_and_dags'/r['continual_wave']/f"{r['model']}__{r['arm']}__{r['task_id']}"
        d=json.loads((ws/'decision_provenance_dag.json').read_text());chain=d['decision_dag']
        k=(r['model'],r['arm']);reprice=0
        for rec in chain:
            typ=rec['decision_type'];delta=float(rec['cost_ledger_delta']['measurement_delta'])
            if typ=='D6_LIVE_GPT_PROVIDER_ERROR':
                provider_errors[rec['chosen_option'].get('error','')]+=1
                provider_by_arm['|'.join(k)]+=1
            if typ=='D0_TASK_ARRIVAL_AND_PILOT_ACQUISITION':
                n=rec['chosen_option']['acquired_pilot_rows']
                # equal 1.5 row rate, same original observations/actions
                reprice+=n*1.5-delta
            elif typ=='D7_D8_CANDIDATE_FIT_AND_FRESH_CONFIRMATION' and delta>0:
                # Original confirm suite always 3 rows for in-scope/OOS cases in 240 suite.
                reprice+=3*1.5-delta
        pricing[k]['deltaM']+=reprice;pricing[k]['C_original']+=r['composite_cost']
        pricing[k]['C_equal_1p5']+=r['composite_cost']+reprice
        pricing[k]['M_equal_1p5']+=r['measurement_cost']+reprice
        if r['is_out_of_scope']:
            for rec in chain:
                if rec['decision_type']=='D6_LIVE_GPT_TOOL_DECISION':oos_decisions[rec['chosen_option']['tool_name']]+=1
        if not r['strict_pass']:
            errors=[x['chosen_option'].get('error') if x['decision_type']=='D6_LIVE_GPT_PROVIDER_ERROR' else x['rejection_reasons'] for x in chain if x['decision_type'] in {'D7_PYTHON_SUBPROCESS_EXECUTION_ERROR','D6_LIVE_GPT_PROVIDER_ERROR'}]
            decisions=[{'turn':x['evidence_inputs'].get('turn'),'action':x['chosen_option'].get('tool_name'),'rationale':x['rationale_summary']} for x in chain if x['decision_type']=='D6_LIVE_GPT_TOOL_DECISION']
            confirms=[x['candidates_evaluated'][0] for x in chain if x['decision_type']=='D7_D8_CANDIDATE_FIT_AND_FRESH_CONFIRMATION']
            types=Counter(x['decision_type'] for x in chain)
            why=('false_confirmation' if r['false_confirmation'] else 'provider_errors' if types['D6_LIVE_GPT_PROVIDER_ERROR'] and not r['evaluated_candidate_count'] else 'execution_errors' if types['D7_PYTHON_SUBPROCESS_EXECUTION_ERROR'] else 'confirmation_refuted' if confirms else 'direct_abstain')
            patterns[(r['model'],r['arm'],why)]+=1
            cases.append({**r,'failure_bucket':why,'errors':errors,'decisions':decisions,'confirmations':confirms,'relative_workspace':str(ws.relative_to(ROOT))})
    result['by_family']={'|'.join(k):summarize(v) for k,v in fam.items()}
    result['by_wave']={'|'.join(k):summarize(v) for k,v in wave.items()}
    result['failure_counts']={'|'.join(k):v for k,v in patterns.items()}
    result['oos_actions']=dict(oos_decisions)
    result['all_v18_4_provider_errors']=dict(provider_errors)
    result['provider_error_records_by_model_arm']=dict(provider_by_arm)
    result['paired_outcomes']={}
    def pair_counts(ps):
        return dict(n=len(ps),both_pass=sum(h['strict_pass'] and b['strict_pass'] for h,b in ps),
            hegel_only=sum(h['strict_pass'] and not b['strict_pass'] for h,b in ps),
            baseline_only=sum(not h['strict_pass'] and b['strict_pass'] for h,b in ps),
            neither=sum(not h['strict_pass'] and not b['strict_pass'] for h,b in ps))
    for model in sorted({r['model'] for r in rows}):
        keyed={(r['task_id'],r['arm']):r for r in rows if r['model']==model and not r['is_out_of_scope']}
        for baseline in ('EqualAuth_Flat_LiveGPT','Strong_Ordinary_LiveGPT'):
            pairs=[(r,keyed[(tid,baseline)]) for (tid,arm),r in keyed.items() if arm=='Hegel_M5_LiveGPT']
            result['paired_outcomes'][model+'|'+baseline]={'all':pair_counts(pairs),
                'both_submitted_diagnostic_only':pair_counts([(h,b) for h,b in pairs if h['evaluated_candidate_count'] and b['evaluated_candidate_count']])}
    result['repricing']={'|'.join(k):{k2:round(v2,5) for k2,v2 in v.items()} for k,v in pricing.items()}
    result['failure_cases']=cases
    result['total_fresh_live_episodes']=len(allrows)
    result['unique_v18_4_episode_keys']=len({(r['model'],r['arm'],r['task_id']) for r in rows})
    (OUT/'v18_recomputed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    cols=['version','continual_wave','model','arm','task_id','status','strict_pass','false_confirmation','api_calls','evaluated_candidate_count','confirm_rmse','heldout_rmse','measurement_cost','compute_cost','composite_cost','effective_billed_tokens']
    with (OUT/'v18_all_episode_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    print(json.dumps({k:result[k] for k in ['sha','chains','total_fresh_live_episodes','unique_v18_4_episode_keys','failure_counts','oos_actions','repricing']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
