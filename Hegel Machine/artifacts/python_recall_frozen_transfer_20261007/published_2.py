import json, math

def select_actions(public_spec, records, memory):
    prior=memory['research_roles']['predictor']
    st=json.load(open('research_state.json'))
    tested,ops={},[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('operator'): ops.append(x)
            for k in ('operators','after_observation_results'):
                for y in x.get(k,[]) or []: walk(y)
        elif isinstance(x,list):
            for y in x: walk(y)
    for s in st.get('method_results',[]):
        p,f=s.get('proposal') or {},s.get('feedback') or {}
        a=p.get('arguments') or {}; m=p.get('model') or (a.get('model') if p.get('action') in ('evaluate','finish') else None)
        if m and isinstance(f.get('fresh_confirmation'),dict): tested[m]=f['fresh_confirmation']
        walk(f)
    for x in st.get('operator_results',[]): walk(x)
    def path(x): return (x or {}).get('path') or (x or {}).get('predictor')
    def confirm(m,lesson): return {'model':m,'lesson':lesson}
    def run(op,**kw): return {'action':'run','arguments':dict(operator=op,**kw)}
    def query(actions,after=None):
        q={'actions':actions}
        if after: q['after_observation']=after
        return {'action':'query','arguments':q}
    def latest(name): return next((x for x in reversed(ops) if x.get('operator')==name),None)
    def meter_round():
        n=len(records); ids=[n,n+1,n+2]; base='shared/meter_round_%d'%n
        seqs=[]
        for seed in (11+n,29+n*2,47+n*3):
            seq=[{'u':max(-1.,min(1.,math.sin(seed*1.71+i*2.399)+.13*math.sin(i*4.13)))} for i in range(24)]
            seqs.append({'kind':'trajectory','sequence':seq,'channels':['meter']})
        return query(seqs,[{'operator':'instrument','path':base+'_instrument.json','records':ids[:2]}, {'operator':'assess_instrument','instrument':base+'_instrument.json','records':[ids[2]]}, {'operator':'response','instrument':base+'_instrument.json','output_dir':base+'_response','records':ids,'validation_records':[ids[2]]}])
    if prior not in tested: return confirm(prior,'Freshly test the unchanged inherited predictor before choosing repair.')
    if tested[prior].get('passed'): return confirm(prior,'Fresh confirmation supports reuse of the unchanged prior.')
    refits=[x for x in ops if x.get('operator')=='refit' and path(x)]
    general=next((x for x in refits if 'post_rejection' not in path(x)),None)
    if general is None:
        n=len(records); return run('refit',model=prior,path='shared/refits/general',records=list(range(n)),validation_records=[n-1] if n else [])
    gp=path(general); gv=(general.get('validation') or {}).get('mse')
    if gv is not None and gv<=1e-10:
        if gp not in tested: return confirm(gp,'Freshly confirm the separately validated meter-free refit.')
        if tested[gp].get('passed'): return confirm(gp,'The validated meter-free refit passed fresh confirmation.')
        rejected=tested[gp]; ix=rejected.get('observation_indices') or []; lo=min(ix) if ix else None; hi=max(ix) if ix else None
        scoped=[x for x in refits if 'post_rejection' in path(x)]
        fresh=[i for i,r in enumerate(records) if r.get('row_indices') and hi is not None and min(r['row_indices'])>hi]
        if not scoped and not fresh:
            seq=[{'u':max(-1.,min(1.,math.sin(43*1.71+i*2.399)+.13*math.sin(i*4.13)))} for i in range(24)]
            return query([{'kind':'trajectory','sequence':seq}])
        if not scoped and fresh:
            val=fresh[-1]; eligible=[i for i,r in enumerate(records) if r.get('row_indices') and lo is not None and min(r['row_indices'])>=lo and i!=val]
            if eligible: return run('refit',model=prior,path='shared/refits/post_rejection',records=eligible+[val],validation_records=[val])
        if scoped:
            sr=scoped[-1]; sp=path(sr); sv=(sr.get('validation') or {}).get('mse')
            if sv is not None and sv<=1e-10:
                if sp not in tested: return confirm(sp,'Confirm post-rejection meter-free refit independently validated on a later trajectory.')
                if tested[sp].get('passed'): return confirm(sp,'Recovered with a post-rejection validated meter-free refit.')
        # If scoped recovery is not validated or fails, investigate measured response.
    ins,ass,resp=latest('instrument'),latest('assess_instrument'),latest('response')
    measured=[i for i,r in enumerate(records) if 'meter' in ((r.get('action') or {}).get('channels') or [])]
    supported=ass and 'meter' in (ass.get('supported_fields') or [])
    if not (ins and supported and resp and len(measured)>=3): return meter_round()
    candidates=resp.get('candidates') or resp.get('candidate_validation') or []
    for c in sorted(candidates,key=lambda z:z.get('validation_mse',float('inf'))):
        m=c.get('path')
        if m and c.get('validation_mse') is not None and c['validation_mse']<=1e-10:
            if m not in tested: return confirm(m,'Freshly confirm the response validated with a supported measured instrument.')
            if tested[m].get('passed'): return confirm(m,'The meter-supported response passed fresh confirmation.')
    rounds=sum(1 for x in ops if x.get('operator')=='instrument')
    if rounds<3: return meter_round()
    return {'stop':'No independently validated candidate remains after the available evidence rounds.'}
