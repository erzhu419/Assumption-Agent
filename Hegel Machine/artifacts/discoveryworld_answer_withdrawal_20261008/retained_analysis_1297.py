def answer(public_spec, observations, memory):
    r = next((x for x in observations if x.get('index') == 429), None)
    if r is None:
        return {'status':'inconclusive','answer':'No record of the 1297-Hz test is available.','evidence':[]}
    msg = str(r.get('lastActionMessage') or r.get('observation', {}).get('lastActionMessage', ''))
    objs = r.get('public_objects') or r.get('observation', {}).get('public_objects', [])
    reactor = next((o for o in objs if o.get('uuid') == 10581), {})
    crystal = next((o for o in objs if o.get('uuid') == 17557), {})
    state = (str(reactor.get('name','')) + ' ' + str(reactor.get('description',''))).lower()
    placed = 'in crystal reactor' in str(crystal).lower() and '10581' in str(crystal)
    if '1297 Hertz' not in msg or not placed:
        return {'status':'inconclusive','answer':'Record 429 does not jointly verify 1297 Hz and crystal 3 inside reactor 3.','evidence':[429]}
    if 'uncalibrated' in state:
        return {'status':'refuted','answer':'Record 429 verifies reactor 3 at 1297 Hz with crystal 3 inside, while the public reactor state is uncalibrated. This refutes the predicted calibrated/activated outcome at 1297 Hz; it does not establish the exact correct frequency or exclude another unmet condition.','evidence':[429]}
    if 'activated' in state or 'calibrated' in state:
        return {'status':'supported','answer':'Record 429 verifies reactor 3 at 1297 Hz with crystal 3 inside and a publicly calibrated/activated state. This supports the tested setting, not its uniqueness.','evidence':[429]}
    return {'status':'inconclusive','answer':'The setpoint and crystal placement are verified, but the public reactor description does not classify calibration.','evidence':[429]}