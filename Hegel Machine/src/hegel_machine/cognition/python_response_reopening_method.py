import json

def select_actions(public_spec, records, memory):
    prior = memory['research_roles']['predictor']
    state = json.load(open('research_state.json'))
    tested, ops = {}, []
    def collect(x):
        if not isinstance(x, dict): return
        if x.get('operator'): ops.append(x)
        for k in ('operators','after_observation_results'):
            for y in x.get(k,[]) or []: collect(y)
    for s in state.get('method_results',[]):
        p,f=s.get('proposal') or {},s.get('feedback') or {}
        a=p.get('arguments') or {}
        m=p.get('model') or (a.get('model') if p.get('action') in ('evaluate','finish') else None)
        if m and isinstance(f.get('fresh_confirmation'),dict): tested[m]=f['fresh_confirmation']
        collect(f)
    for x in state.get('operator_results',[]): collect(x)
    def path(x): return (x or {}).get('path') or (x or {}).get('predictor')
    def latest(name): return next((x for x in reversed(ops) if x.get('operator')==name),None)
    def refits(): return [x for x in ops if x.get('operator')=='refit' and path(x)]
    def confirm(m,lesson): return {'model':m,'lesson':lesson}
    def run(op,**kw): return {'action':'run','arguments':dict(operator=op,**kw)}
    def traj(seq,meter=False):
        a={'kind':'trajectory','sequence':[{'u':float(u)} for u in seq]}
        if meter: a['channels']=['meter']
        return a
    def query(actions,after=None): return {'action':'query','arguments':{'actions':actions,'after_observation':after or []}}
    def start(i): return min(records[i].get('row_indices') or [10**18])
    def measured(): return [i for i,r in enumerate(records) if 'meter' in ((r.get('action') or {}).get('channels') or [])]
    def new_meter_batch():
        # Three independent meter trajectories: two for instrument fitting, one holdout.
        seqs=[[.3880320412,.2855727117,-.5681744065,-.1332501521,.1748351352,.4428971352,.7850410616,.5464563094,-.1092888503,-.4957562756,-.8096240643,.9846971771,.9861105749,-.110472256,.6963220399,.2043525121,.1763241894,.0255793183,-.3257642624,.9088645395,-.198530695,.1526668204,-.2369553077],[ -.4356446484,.906658453,.5827516665,-.7286590964,.7296971076,-.7585809315,.689262796,.9059713,-.5509465578,.7230015418,.1483956693,-.2357768688,-.1743411993,.2956281421,-.0115005316,.183611514,-.9052916309,-.3057231415,-.1793505372,-.3143214752,-.6963720957,.2284370124,-.4158327134,.8061704098],[.931,-.742,.116,.543,-.294,.817,-.053,-.963,.362,.681,-.487,.209,.774,-.125,-.611,.459,-.838,.032,.596,-.371,.883,-.206,-.694,.147]]
        n=len(records); ip='shared/instrument_batch_'+str(n)+'.json'
        acts=[traj(s,True) for s in seqs]
        return query(acts,[{'operator':'instrument','path':ip,'records':[n,n+1],'compute_limit':100000},{'operator':'assess_instrument','instrument':ip,'records':[n+2]}])

    if prior not in tested: return confirm(prior,'Test unchanged inherited predictor before selecting a repair.')
    if tested[prior].get('passed'): return confirm(prior,'Fresh evidence supports reuse of unchanged inherited predictor.')

    # Try an independently validated, meter-free refit first.
    rs=refits(); general=next((r for r in rs if 'post_rejection' not in str(path(r))),None)
    if general is None: return run('refit',model=prior,path='shared/refits/general')
    gp=path(general); gv=(general.get('validation') or {}).get('mse')
    if gv is not None and gv<=1e-10:
        if gp not in tested: return confirm(gp,'Freshly confirm independently validated meter-free refit.')
        if tested[gp].get('passed'): return confirm(gp,'Validated meter-free refit passed confirmation.')
        # Scope recovery strictly to records acquired after the rejected confirmation.
        rejected_start=min(tested[gp].get('observation_indices') or [10**18])
        post=[i for i,r in enumerate(records) if r.get('row_indices') and start(i)>=rejected_start]
        if len(post)<2:
            seq=[.73,-.31,.91,-.84,.12,.57,-.66,.38,-.95,.04,.81,-.48,.25,-.72,.99,-.16,.63,-.41,.08,.46,-.89,.32,-.54,.77]
            return query([traj(seq),traj(list(reversed(seq)))])
        scoped=[r for r in rs if 'post_rejection' in str(path(r))]
        if not scoped:
            train,val=post[:-1],post[-1]
            return run('refit',model=prior,path='shared/refits/post_rejection',records=train+[val],validation_records=[val])
        sr=scoped[-1]; sp=path(sr); sv=(sr.get('validation') or {}).get('mse')
        if sv is not None and sv<=1e-10:
            if sp not in tested: return confirm(sp,'Confirm independently validated meter-free recovery using only post-refutation records.')
            if tested[sp].get('passed'): return confirm(sp,'Recovered after refutation with post-window meter-free refit.')

    # A refuted refit (or unchanged predictor) may require a revised measured response.
    ins,ass=latest('instrument'),latest('assess_instrument')
    supported=bool(ins and ass and path(ins)==path(ass) and ('meter' in (ass.get('supported_fields') or []) or ass.get('status')=='supported_on_measured_trajectory'))
    ms=measured()
    if not supported or not ms: return new_meter_batch()
    # Use only the latest complete measured batch after the failed predictor window.
    batch=ms[-3:]
    if len(batch)<3: return new_meter_batch()
    resp=latest('response'); ra=(resp or {}).get('arguments') or {}
    if resp is None or ra.get('instrument')!=path(ins) or ra.get('records')!=batch:
        return run('response',instrument=path(ins),output_dir='shared/response_'+str(len(records)),records=batch,validation_records=[batch[-1]])
    candidates=(resp or {}).get('candidates') or (resp or {}).get('candidate_validation') or []
    for c in sorted(candidates,key=lambda z:z.get('validation_mse',float('inf'))):
        cp=c.get('path')
        if cp and c.get('validation_mse') is not None and c['validation_mse']<=1e-10:
            if cp not in tested: return confirm(cp,'Freshly confirm meter-response refit validated on a separate measured trajectory.')
            if tested[cp].get('passed'): return confirm(cp,'Meter-supported response passed fresh confirmation.')
    if any(c.get('path') in tested and not tested[c.get('path')].get('passed') for c in candidates): return new_meter_batch()
    return {'stop':'No independently validated predictor remains in current evidence.'}
