import json

def select_actions(public_spec, records, memory):
    prior = memory['research_roles']['predictor']
    state = json.load(open('research_state.json'))
    tested, ops = {}, []

    def add(r):
        if isinstance(r, dict) and r.get('operator'):
            ops.append(r)
        if isinstance(r, dict):
            for key in ('operators', 'after_observation_results'):
                for x in r.get(key, []) or []:
                    add(x)

    for step in state.get('method_results', []):
        p, f = step.get('proposal') or {}, step.get('feedback') or {}
        a = p.get('arguments') or {}
        model = p.get('model') or (a.get('model') if p.get('action') in ('evaluate', 'finish') else None)
        if model and isinstance(f.get('fresh_confirmation'), dict):
            tested[model] = f['fresh_confirmation']
        add(f)
    for r in state.get('operator_results', []):
        add(r)

    def path(r):
        return (r or {}).get('path') or (r or {}).get('predictor')
    def confirm(model, lesson):
        return {'model': model, 'lesson': lesson}
    def latest(name):
        return next((r for r in reversed(ops) if r.get('operator') == name), None)
    def refits():
        return [r for r in ops if r.get('operator') == 'refit' and path(r)]

    if prior not in tested:
        return confirm(prior, 'Freshly test the unchanged inherited predictor before deciding whether revision is needed.')
    if tested[prior].get('passed'):
        return confirm(prior, 'Fresh confirmation supports reuse of the unchanged inherited predictor.')

    rs = refits()
    general = next((r for r in rs if 'post_rejection' not in path(r)), None)
    if general is None:
        return {'action': 'run', 'arguments': {'operator': 'refit', 'model': prior, 'path': 'shared/refits/general'}}
    gp = path(general)
    gv = (general.get('validation') or {}).get('mse')
    if gv is not None and gv <= 1e-10:
        if gp not in tested:
            return confirm(gp, 'Freshly confirm the separately trajectory-validated refit without meter readings.')
        if tested[gp].get('passed'):
            return confirm(gp, 'The independently validated meter-free refit passed fresh confirmation.')
        rejected = tested[gp]
        zero_meter_route = rejected.get('mse', 1.0) <= 0.012
        if zero_meter_route:
            rejected_rows = rejected.get('observation_indices') or []
            end = max(rejected_rows) if rejected_rows else -1
            post = [i for i, rec in enumerate(records)
                    if rec.get('row_indices') and min(rec['row_indices']) > end]
            scoped = [r for r in rs if 'post_rejection' in path(r)]
            if len(post) < 2:
                seq = [.73,-.31,.91,-.84,.12,.57,-.66,.38,-.95,.04,.81,-.48,.25,-.72,.99,-.16,.63,-.41,.08,.46,-.89,.32,-.54,.77]
                acts = [{'kind': 'trajectory', 'sequence': [{'u': x} for x in q]} for q in (seq, list(reversed(seq)))]
                return {'action': 'query', 'arguments': {'actions': acts}}
            if not scoped:
                chosen = post[-2:]
                return {'action': 'run', 'arguments': {'operator': 'refit', 'model': prior, 'path': 'shared/refits/post_rejection', 'records': chosen, 'validation_records': [chosen[-1]]}}
            sr = scoped[-1]
            sp = path(sr)
            sv = (sr.get('validation') or {}).get('mse')
            if sv is not None and sv <= 1e-10:
                if sp not in tested:
                    return confirm(sp, 'Confirm recovery using only post-refutation records, including a separately validated new trajectory, without meter readings.')
                if tested[sp].get('passed'):
                    return confirm(sp, 'Recovered from refutation with an independently validated, post-refutation meter-free refit.')

    ins, ass = latest('instrument'), latest('assess_instrument')
    supported = bool(ass and ('meter' in (ass.get('supported_fields') or []) or ass.get('status') == 'supported_on_measured_trajectory'))
    measured = [i for i, r in enumerate(records) if 'meter' in (r.get('action') or {}).get('channels', [])]
    if not (ins and supported):
        n = len(records)
        seqs = [[.3880320412,.2855727117,-.5681744065,-.1332501521,.1748353465,.4428971352,.7850410616,.5464563094,-.1092888503,-.4957562756,-.8096240643,.9846971771,.9861105749,-.110472256,.6963220399,.2043525121,.1763241894,.0255793183,-.3257642624,.9088645395,-.198530695,.1526668204,-.2369553077],[-.4356446484,.906658453,.5827516665,-.7286590964,.7296971076,-.7585809315,.689262796,.9059713,-.5509465578,.7230015418,.1483956693,-.2357768688,-.1743411993,.2956281421,-.0115005316,.183611514,-.9052916309,-.3057231415,-.1793505372,-.3143214752,-.6963720957,.2284370124,-.4158327134,.8061704098],[.931,-.742,.116,.543,-.294,.817,-.053,-.963,.362,.681,-.487,.209,.774,-.125,-.611,.459,-.838,.032,.596,-.371,.883,-.206,-.694,.147]]
        actions = [{'kind': 'trajectory', 'sequence': [{'u': x} for x in q], 'channels': ['meter']} for q in seqs]
        return {'action': 'query', 'arguments': {'actions': actions, 'after_observation': [{'operator': 'instrument', 'path': 'shared/research_instrument.json', 'records': [n, n+1], 'compute_limit': 100000}, {'operator': 'assess_instrument', 'instrument': 'shared/research_instrument.json', 'records': [n+2]}]}}

    if latest('response') is None:
        return {'action': 'run', 'arguments': {'operator': 'response', 'instrument': path(ins) or 'shared/research_instrument.json', 'output_dir': 'shared/revised', 'records': measured, 'validation_records': [measured[-1]]}}
    response = latest('response')
    candidates = response.get('candidates') or response.get('candidate_validation') or []
    for c in sorted(candidates, key=lambda x: x.get('validation_mse', float('inf'))):
        cp = c.get('path')
        if cp and c.get('validation_mse') is not None and c['validation_mse'] <= 1e-10 and cp not in tested:
            return confirm(cp, 'Recover after refit rejection with a meter-supported response independently validated on another measured trajectory.')
    return {'stop': 'No independently trajectory-validated response candidate is available.'}
