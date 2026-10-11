def select_actions(public_spec, records, memory):
    import json
    prior=memory.get('research_roles',{}).get('predictor','shared/revised/response_00.py')
    try:
        with open('research_state.json') as f: state=json.load(f)
    except Exception: state={}
    methods=state.get('method_results',[])
    ops=state.get('operator_results',[])
    # Test unchanged inherited predictor first; passing fresh confirmation is sufficient reuse evidence.
    if not methods:
        return {'model':prior,'lesson':'Attempt unchanged prior first; retain it only on fresh predictive confirmation. On failure acquire current measured evidence, check instrument consistency and whole-trajectory response validation, then require fresh confirmation of a repair.'}
    conf=methods[-1].get('feedback',{}).get('fresh_confirmation',{})
    if conf.get('passed'):
        return {'model':methods[-1].get('proposal',{}).get('model',prior),'lesson':'Unchanged inherited predictor passed fresh predictive confirmation; preserve it without unnecessary refitting. If it fails in another apparatus, assess current paid instrument readings and trajectory-held-out response candidates, then require fresh confirmation.'}
    # Failed prior confirmation: collect diverse reset trajectories with readings.
    if not any(o.get('operator')=='response' for o in ops):
        n=min(24,int(public_spec.get('max_probe_length',24)))
        pats=[[-1,.25,1,-.5,0,.75,-1,.5,.25,-1,1,0,-.75,.5,-1,1,.25,-.5,0,1,-.25,.75,-1,.5],[1,-.25,-1,.5,0,-.75,1,-.5,-.25,1,-1,0,.75,-.5,1,-1,-.25,.5,0,-1,.25,-.75,1,-.5]]
        ch=['meter'] if any(c.get('name')=='meter' for c in public_spec.get('channels',[])) else []
        actions=[{'kind':'trajectory','sequence':[{'u':x} for x in p[:n]],'channels':ch} for p in pats]
        return {'actions':actions,'after_observation':[{'operator':'instrument','path':'shared/current_meter.json'},{'operator':'response','instrument':'shared/current_meter.json','output_dir':'shared/current_candidates'}]}
    resp=next((o for o in reversed(ops) if o.get('operator')=='response' and o.get('candidates')),None)
    if resp:
        diag=resp.get('paid_meter_consistency',{})
        consistent=diag.get('status')=='available' and any(h.get('mse',1e99)<1e-8 for h in diag.get('hypotheses',[]))
        valid=[c for c in resp['candidates'] if c.get('validation_mse',1e99)<1e-7]
        if consistent and valid:
            best=min(valid,key=lambda c:c['validation_mse'])
            return {'model':best['path'],'lesson':'Prior failed fresh confirmation. Current meter readings agree with a fitted instrument hypothesis and the repaired response validates on a held-out whole trajectory; fresh confirmation determines adoption.'}
    return {'actions':[],'after_observation':[{'operator':'instrument','path':'shared/current_meter.json'},{'operator':'response','instrument':'shared/current_meter.json','output_dir':'shared/current_candidates'}]}