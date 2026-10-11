"""Reuse prior predictors when confirmed; validate every repair on held-out trajectories."""
import json

def select_actions(public_spec, records, memory):
    prior = memory['research_roles']['predictor']
    state = json.load(open('research_state.json'))
    tested, ops = {}, []
    def add(x):
        if isinstance(x, dict):
            if x.get('operator'): ops.append(x)
            for k in ('operators','after_observation_results'):
                for y in x.get(k,[]) or []: add(y)
    for s in state.get('method_results',[]):
        p,f=s.get('proposal') or {},s.get('feedback') or {}
        a=p.get('arguments') or {}
        m=p.get('model') or (a.get('model') if p.get('action') in ('evaluate','finish') else None)
        if m and isinstance(f.get('fresh_confirmation'),dict): tested[m]=f['fresh_confirmation']
        add(f)
    for x in state.get('operator_results',[]): add(x)
    def path(x): return (x or {}).get('path') or (x or {}).get('predictor')
    def confirm(m,why): return {'model':m,'lesson':why}
    def latest(name): return next((x for x in reversed(ops) if x.get('operator')==name),None)
    def refits(): return [x for x in ops if x.get('operator')=='refit' and path(x)]
    def query(actions,after=None):
        q={'actions':actions}
        if after: q['after_observation']=after
        return {'action':'query','arguments':q}
    def run(op,**kw): return {'action':'run','arguments':dict(operator=op,**kw)}
    def trajectories(seed,count=3,meter=True):
        import math
        out=[]
        for j in range(count):
            seq=[{'u':max(-1.,min(1.,math.sin((seed+j*13)*1.71+i*2.399)+.13*math.sin(i*4.13+j)))} for i in range(24)]
            out.append({'kind':'trajectory','sequence':seq,'channels':['meter'] if meter else []})
        return out
    if prior not in tested: return confirm(prior,'Test unchanged earlier predictor before considering repairs.')
    if tested[prior].get('passed'): return confirm(prior,'Fresh confirmation supports unchanged-prior reuse.')
    rs=refits()
    general=next((r for r in rs if 'post_rejection' not in path(r)),None)
    if general is None:
        n=len(records)
        return run('refit',model=prior,path='shared/refits/general',records=list(range(n)),validation_records=[n-1] if n else [])
    gp=path(general); gv=(general.get('validation') or {}).get('mse')
    if gv is not None and gv<=1e-10:
        if gp not in tested: return confirm(gp,'Confirm separately trajectory-validated refit without meter readings.')
        if tested[gp].get('passed'): return confirm(gp,'Validated meter-free refit passed fresh confirmation.')
        rej=tested[gp]; ix=rej.get('observation_indices') or []
        if ix:
            hi=max(ix); later=[i for i,r in enumerate(records) if r.get('row_indices') and min(r['row_indices'])>hi]
            scoped=[r for r in rs if 'post_rejection' in path(r)]
            if not later: return query(trajectories(47+len(records)*3,2,False))
            val=later[-1]
            if not scoped:
                train=[i for i,r in enumerate(records) if i!=val and r.get('row_indices') and min(r['row_indices'])>hi]
                if train: return run('refit',model=prior,path='shared/refits/post_rejection',records=train+[val],validation_records=[val])
            else:
                sr=scoped[-1]; sp=path(sr); sv=(sr.get('validation') or {}).get('mse')
                if sv is not None and sv<=1e-10:
                    if sp not in tested: return confirm(sp,'Confirm post-rejection meter-free refit using only later records and an independent later validation trajectory.')
                    if tested[sp].get('passed'): return confirm(sp,'Recovered with validated post-rejection meter-free refit.')
    # If refitting does not recover, learn and validate a measured-input response.
    ins,ass,resp=latest('instrument'),latest('assess_instrument'),latest('response')
    measured=[i for i,r in enumerate(records) if 'meter' in ((r.get('action') or {}).get('channels') or [])]
    supported=bool(ass and 'meter' in (ass.get('supported_fields') or []))
    if not (ins and supported and len(measured)>=3):
        n=len(records); new=[n,n+1,n+2]; base='shared/measured_repair_%d'%n
        return query(trajectories(11+n*3),[{'operator':'instrument','path':base+'_instrument.json','records':new[:2]}, {'operator':'assess_instrument','instrument':base+'_instrument.json','records':[new[2]]}, {'operator':'response','instrument':base+'_instrument.json','output_dir':base+'_response','records':new,'validation_records':[new[2]]}])
    if resp is None:
        base=path(ins) or 'shared/research_instrument.json'
        return run('response',instrument=base,output_dir='shared/revised',records=measured,validation_records=[measured[-1]])
    candidates=resp.get('candidates') or resp.get('candidate_validation') or []
    for c in sorted(candidates,key=lambda z:z.get('validation_mse',float('inf'))):
        m=c.get('path'); err=c.get('validation_mse')
        if m and err is not None and err<=1e-10:
            if m not in tested: return confirm(m,'Freshly confirm response independently validated with supported meter calibration.')
            if tested[m].get('passed'): return confirm(m,'Meter-supported response passed fresh confirmation.')
    # A measured response also failed: acquire a new post-counterexample measured block,
    # refit calibration and response on that block, and independently validate its last trajectory.
    failed=[v for v in tested.values() if v.get('passed') is False]
    start=max((max(v.get('observation_indices') or [-1]) for v in failed),default=-1)
    recent=[i for i in measured if records[i].get('row_indices') and min(records[i]['row_indices'])>start]
    if len(recent)<3:
        n=len(records); new=[n,n+1,n+2]; base='shared/post_failure_%d'%n
        return query(trajectories(29+n*5),[{'operator':'instrument','path':base+'_instrument.json','records':new[:2]}, {'operator':'assess_instrument','instrument':base+'_instrument.json','records':[new[2]]}, {'operator':'response','instrument':base+'_instrument.json','output_dir':base+'_response','records':new,'validation_records':[new[2]]}])
    return {'stop':'No independently validated candidate currently supports fresh confirmation.'}
