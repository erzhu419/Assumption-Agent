import json

def select_actions(spec, records, memory):
    prior = memory['research_roles']['predictor']
    state = json.load(open('research_state.json'))
    steps = state.get('method_results', [])
    if not steps:
        return {'model': prior, 'lesson': 'Check applicability by fresh confirmation of the unchanged inherited predictor first.'}

    prior_failed = False
    refit_result = None
    instrument_result = None
    assessment_result = None
    response_result = None
    tested = {}
    for step in steps:
        proposal = step.get('proposal') or {}
        feedback = step.get('feedback') or {}
        args = proposal.get('arguments') or {}
        model = proposal.get('model') or (args.get('model') if proposal.get('action') in ('evaluate', 'finish') else None)
        if model:
            confirmation = feedback.get('fresh_confirmation') or {}
            if confirmation: tested[model] = confirmation
            if model == prior and confirmation.get('passed') is False: prior_failed = True
        results = feedback.get('after_observation_results') or []
        candidates = [feedback] + results
        for result in candidates:
            if not isinstance(result, dict): continue
            op = result.get('operator')
            if op == 'refit': refit_result = result
            elif op == 'instrument': instrument_result = result
            elif op == 'assess_instrument': assessment_result = result
            elif op == 'response': response_result = result

    if not prior_failed:
        return {'model': prior, 'lesson': 'Reuse the unchanged prior when fresh confirmation passes.'}

    if refit_result is None:
        return {'action': 'run', 'arguments': {'operator': 'refit', 'model': prior, 'path': 'shared/refits/revision'}}
    refit_path = refit_result.get('path') or refit_result.get('predictor')
    refit_mse = ((refit_result.get('validation') or {}).get('mse'))
    if refit_path and refit_mse is not None and refit_mse <= 1e-10:
        return {'model': refit_path, 'lesson': 'After prior failure, use a separately trajectory-validated refit and confirm it freshly.'}

    if instrument_result is None or assessment_result is None:
        # Two measured trajectories distinguish instrument dynamics; fit on the first and assess on the second.
        seq1 = [0.3880320412,0.2855727117,-0.5681744065,-0.1332501521,0.1748353465,0.4428971352,0.7850410616,0.5464563094,-0.1092888503,-0.4957562756,-0.8096240643,0.9846971771,0.9861105749,-0.1104722560,0.6963220399,0.4261743569,0.2043525121,0.1763241894,0.0255793183,-0.3257642624,0.9088645395,-0.1985306950,0.1526668204,-0.2369553077]
        seq2 = [-0.4356446484,0.9066584530,0.5827516665,-0.7286590964,0.7296971076,-0.7585809315,0.6892627960,0.9059713000,-0.5509465578,0.7230015418,0.1483956693,-0.2357768688,-0.1743411993,0.2956281421,-0.0115005316,0.1836115140,-0.9052916309,-0.3057231415,-0.1793505372,-0.3143214752,-0.6963720957,0.2284370124,-0.4158327134,0.8061704098]
        return {'action':'query','arguments':{'actions':[{'kind':'trajectory','sequence':[{'u':x} for x in seq1],'channels':['meter']},{'kind':'trajectory','sequence':[{'u':x} for x in seq2],'channels':['meter']}], 'after_observation':[{'operator':'instrument','path':'shared/research_instrument.json','records':[4],'compute_limit':100000},{'operator':'assess_instrument','instrument':'shared/research_instrument.json','records':[5]}]}}

    ip = instrument_result.get('path') or 'shared/research_instrument.json'
    if response_result is None:
        return {'action':'run','arguments':{'operator':'response','instrument':ip,'output_dir':'shared/revised'}}
    candidates = response_result.get('candidates') or []
    viable = [c for c in candidates if c.get('path') and c.get('validation_mse') is not None and c.get('validation_mse') <= 1e-10]
    viable.sort(key=lambda c: c['validation_mse'])
    for candidate in viable:
        path = candidate['path']
        if path not in tested:
            return {'model':path,'lesson':'When refitting fails, learn and validate the measured instrument, then freshly confirm its validated response model.'}
    return {'stop':'No revised candidate is supported by separate paid trajectories.'}
