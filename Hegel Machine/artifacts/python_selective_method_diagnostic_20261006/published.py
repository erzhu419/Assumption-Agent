import json

def select_actions(spec, records, memory):
    prior = memory['research_roles']['predictor']
    state = json.load(open('research_state.json'))
    steps = state.get('method_results', [])
    if not steps:
        return {'model': prior, 'lesson': 'Freshly confirm the inherited predictor before deciding whether revision is needed.'}

    prior_failed = False
    instrument = None
    response = None
    tested = set()
    for step in steps:
        proposal = step.get('proposal') or {}
        feedback = step.get('feedback') or {}
        model = proposal.get('model')
        if model:
            conf = feedback.get('fresh_confirmation') or {}
            if conf:
                tested.add(model)
                if model == prior and conf.get('passed') is False:
                    prior_failed = True
        results = (feedback.get('after_observation_results') or [])
        for result in [feedback] + results:
            if not isinstance(result, dict):
                continue
            if result.get('operator') == 'instrument':
                instrument = result
            elif result.get('operator') == 'response':
                response = result

    if not prior_failed:
        return {'model': prior, 'lesson': 'Retain the unchanged predictor when fresh confirmation passes.'}

    if instrument is None:
        n = len(records)
        seqs = [
          [0.3880320412,0.2855727117,-0.5681744065,-0.1332501521,0.1748353465,0.4428971352,0.7850410616,0.5464563094,-0.1092888503,-0.4957562756,-0.8096240643,0.9846971771,0.9861105749,-0.1104722560,0.6963220399,0.4261743569,0.2043525121,0.1763241894,0.0255793183,-0.3257642624,0.9088645395,-0.1985306950,0.1526668204,-0.2369553077],
          [-0.4356446484,0.9066584530,0.5827516665,-0.7286590964,0.7296971076,-0.7585809315,0.6892627960,0.9059713000,-0.5509465578,0.7230015418,0.1483956693,-0.2357768688,-0.1743411993,0.2956281421,-0.0115005316,0.1836115140,-0.9052916309,-0.3057231415,-0.1793505372,-0.3143214752,-0.6963720957,0.2284370124,-0.4158327134,0.8061704098],
          [0.918,-0.741,0.106,0.632,-0.253,0.487,-0.964,0.321,0.772,-0.119,-0.588,0.845,-0.397,0.054,0.691,-0.823,0.234,0.956,-0.462,0.137,-0.705,0.378,-0.017,0.559]
        ]
        actions = [{'kind':'trajectory','sequence':[{'u':x} for x in seq], 'channels':['meter']} for seq in seqs]
        return {'action':'query','arguments':{'actions':actions,'after_observation':[
          {'operator':'instrument','path':'shared/research_instrument.json','records':[n,n+1],'compute_limit':100000},
          {'operator':'assess_instrument','instrument':'shared/research_instrument.json','records':[n+2]}
        ]}}

    ip = instrument.get('path') or 'shared/research_instrument.json'
    if response is None:
        n = len(records)
        return {'action':'run','arguments':{'operator':'response','instrument':ip,'output_dir':'shared/revised','records':[n-3,n-2,n-1],'validation_records':[n-1]}}

    candidates = response.get('candidates') or []
    candidates = sorted([c for c in candidates if c.get('path')], key=lambda c: c.get('validation_mse') if c.get('validation_mse') is not None else float('inf'))
    for candidate in candidates:
        if candidate['path'] not in tested:
            return {'model':candidate['path'],'lesson':'After inherited-model failure, fit a response against the measured instrument coordinate and freshly confirm the revised predictor.'}
    return {'stop':'No untested revised response candidate remains.'}
