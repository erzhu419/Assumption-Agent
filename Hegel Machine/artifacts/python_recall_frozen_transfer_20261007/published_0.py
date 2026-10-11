"""Reuse, validate, then repair a temporal predictor from paid evidence."""
import json

def select_actions(public_spec, records, memory):
    prior=memory['research_roles']['predictor']
    state=json.load(open('research_state.json'))
    tested,ops={},[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('operator'): ops.append(x)
            for k in ('operators','after_observation_results'):
                for y in x.get(k,[]) or []: walk(y)
    for s in state.get('method_results',[]):
        p=s.get('proposal') or {}; f=s.get('feedback') or {}
        a=p.get('arguments') or {}
        m=p.get('model') or (a.get('model') if p.get('action') in ('evaluate','finish') else None)
        if m and isinstance(f.get('fresh_confirmation'),dict): tested[m]=f['fresh_confirmation']
        walk(f)
    for x in state.get('operator_results',[]): walk(x)
    def path(x): return (x or {}).get('path') or (x or {}).get('predictor')
    def run(op,**kw): return {'action':'run','arguments':dict(operator=op,**kw)}
    def confirm(m,why): return {'model':m,'lesson':why}
    def query(actions,after=None):
        q={'actions':actions}
        if after: q['after_observation']=after
        return {'action':'query','arguments':q}
    def traj(seed,meter=False):
        import math
        return {'kind':'trajectory','sequence':[{'u':max(-1.,min(1.,math.sin(seed*1.71+i*2.399)+.13*math.sin(i*4.13)))} for i in range(24)],'channels':['meter'] if meter else []}
    if prior not in tested:
        return confirm(prior,'Freshly test the unchanged earlier predictor before choosing reuse or repair.')
    if tested[prior].get('passed'):
        return confirm(prior,'Fresh confirmation supports unchanged-prior reuse.')
    refits=[x for x in ops if x.get('operator')=='refit' and path(x)]
    general=next((x for x in refits if 'post_rejection' not in path(x)),None)
    if general is None:
        n=len(records)
        return run('refit',model=prior,path='shared/refits/general',records=list(range(n)),validation_records=[n-1] if n else [])
    gp=path(general); gv=(general.get('validation') or {}).get('mse')
    if gv is not None and gv<=1e-10:
        if gp not in tested: return confirm(gp,'Freshly confirm the separately trajectory-validated refit without meter readings.')
        if tested[gp].get('passed'): return confirm(gp,'The independently validated, meter-free refit passed fresh confirmation.')
    rejected=tested.get(gp,{})
    # After a refitted candidate is actually rejected, test a fit scoped to
    # the rejected confirmation window and a newly acquired, independent holdout.
    if rejected and not rejected.get('passed'):
        ix=rejected.get('observation_indices') or []
        start=min(ix) if ix else None; end=max(ix) if ix else None
        post=[i for i,r in enumerate(records) if r.get('row_indices') and (start is None or min(r['row_indices'])>=start)]
        fresh=[i for i,r in enumerate(records) if r.get('row_indices') and end is not None and min(r['row_indices'])>end]
        scoped=[x for x in refits if 'post_rejection' in path(x)]
        if not fresh:
            return query([traj(47+len(records)*3)])
        if not scoped:
            val=fresh[-1]; chosen=[i for i in post if i!=val]
            if chosen:
                return run('refit',model=prior,path='shared/refits/post_rejection',records=chosen+[val],validation_records=[val])
        else:
            sr=scoped[-1]; sp=path(sr); sv=(sr.get('validation') or {}).get('mse')
            if sv is not None and sv<=1e-10:
                if sp not in tested: return confirm(sp,'Freshly confirm a post-rejection, meter-free refit with an independent new-trajectory holdout.')
                if tested[sp].get('passed'): return confirm(sp,'Recovered with an independently validated post-rejection meter-free refit.')
    # If trajectory-only fitting is unsupported by its holdout, investigate
    # the auxiliary measurement and fit a response only after instrument support.
    ins=next((x for x in reversed(ops) if x.get('operator')=='instrument'),None)
    ass=next((x for x in reversed(ops) if x.get('operator')=='assess_instrument'),None)
    resp=next((x for x in reversed(ops) if x.get('operator')=='response'),None)
    measured=[i for i,r in enumerate(records) if 'meter' in ((r.get('action') or {}).get('channels') or [])]
    supported=bool(ass and 'meter' in (ass.get('supported_fields') or []))
    if not (ins and supported and len(measured)>=3):
        n=len(records); inds=[n,n+1,n+2]; base='shared/measured_repair_%d'%n
        acts=[traj(11+n*3,True),traj(19+n*5,True),traj(31+n*7,True)]
        after=[{'operator':'instrument','path':base+'_instrument.json','records':inds[:2]},
               {'operator':'assess_instrument','instrument':base+'_instrument.json','records':[inds[2]]},
               {'operator':'response','instrument':base+'_instrument.json','output_dir':base+'_response','records':inds,'validation_records':[inds[2]]}]
        return query(acts,after)
    if not resp:
        return run('response',instrument=path(ins),output_dir='shared/measured_response',records=measured,validation_records=[measured[-1]])
    candidates=resp.get('candidates') or resp.get('candidate_validation') or []
    for c in sorted(candidates,key=lambda z:z.get('validation_mse',float('inf'))):
        m=c.get('path')
        if m and c.get('validation_mse') is not None and c['validation_mse']<=1e-10 and m not in tested:
            return confirm(m,'Freshly confirm the independently validated response using a supported measured instrument.')
        if m in tested and tested[m].get('passed'):
            return confirm(m,'Recovered with a fresh-confirmed response after validating its instrument and response.')
    # A measured response was refuted: acquire a new measured set and reassess.
    n=len(records); inds=[n,n+1,n+2]; base='shared/measured_repair_%d'%n
    return query([traj(53+n,True),traj(67+n,True),traj(79+n,True)],
      [{'operator':'instrument','path':base+'_instrument.json','records':inds[:2]},
       {'operator':'assess_instrument','instrument':base+'_instrument.json','records':[inds[2]]},
       {'operator':'response','instrument':base+'_instrument.json','output_dir':base+'_response','records':inds,'validation_records':[inds[2]]}])
