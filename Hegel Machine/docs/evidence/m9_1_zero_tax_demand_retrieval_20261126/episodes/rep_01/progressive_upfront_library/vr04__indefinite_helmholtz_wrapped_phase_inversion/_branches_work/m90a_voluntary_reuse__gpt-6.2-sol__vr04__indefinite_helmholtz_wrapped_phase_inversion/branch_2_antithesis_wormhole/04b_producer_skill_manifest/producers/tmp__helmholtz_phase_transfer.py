import runpy
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar

ns = runpy.run_path('/tmp/helmholtz_public.py')
m, cases = ns['m'], ns['cases']
for idx, case in enumerate(cases):
    p = case['problem']
    pilot, survey = p['pilot_calibration'], p['survey']
    ref = pilot['known_field_re'] + 1j * pilot['known_field_im']
    phase = np.angle(np.vdot(ref, m._complex_observations(pilot)))
    fields = m._source_fields(p)
    i, j = survey['sensor_ij'].T
    design = fields[i, j]
    data = m._complex_observations(survey) * np.exp(-1j * phase)
    groups = 2*(i >= fields.shape[0]/2)+(j >= fields.shape[1]/2)
    equivalent_phase, loss = defaultdict(list), defaultdict(list)
    for group in np.unique(groups):
        take = groups == group
        train_a = m._robust_sources(design[~take], data[~take], survey['tikhonov_alpha'])
        pred = design[take] @ train_a
        obs = data[take]
        def score(delta):
            return np.median(np.abs(obs - np.exp(1j*delta)*pred))
        fit = minimize_scalar(score, bounds=(-.12, .12), method='bounded')
        equivalent_phase[group] = fit.x
        loss[group] = (score(0), fit.fun)
    print('case', idx, 'quadrant phase corrections (rad) and median residual before/after')
    for group in equivalent_phase:
        print(int(group), round(float(equivalent_phase[group]), 6), np.round(loss[group], 7))
