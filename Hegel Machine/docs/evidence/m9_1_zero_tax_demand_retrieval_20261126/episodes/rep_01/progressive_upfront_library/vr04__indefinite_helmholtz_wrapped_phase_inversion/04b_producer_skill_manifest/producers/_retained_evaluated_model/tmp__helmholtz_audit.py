import runpy
import numpy as np

ns = runpy.run_path('/tmp/helmholtz_public.py')
m, cases = ns['m'], ns['cases']
keys = ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs')
limits = np.array([.03, .02, .03])
worst_pilot, worst_survey = [], []

def errors(p, gt):
    out = m.invert_helmholtz_sources_and_phase(p)
    return np.array([np.linalg.norm(out[keys[0]]-gt[keys[0]])/np.linalg.norm(gt[keys[0]]), abs(np.angle(np.exp(1j*(out[keys[1]]-gt[keys[1]])))), np.linalg.norm(out[keys[2]]-gt[keys[2]])/np.linalg.norm(gt[keys[2]])])

def subset(p, channel, keep):
    return dict(p, **{channel: {k: np.asarray(v)[keep] if np.ndim(v) else v for k,v in p[channel].items()}})

for idx, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
    for rank in np.argsort(abs(ref)):
        worst_pilot.append(errors(subset(p, 'pilot_calibration', np.arange(ref.size) != rank), gt))
    fields = m._source_fields(p)
    i,j = survey['sensor_ij'].T
    d = fields[i,j]
    nx,ny = fields.shape[:2]
    quadrant = 2*(i >= nx/2)+(j >= ny/2)
    clean = d @ gt[keys[0]]
    obs = m._complex_observations(survey)*np.exp(-1j*gt[keys[1]])
    r = obs-clean
    print('case', idx, 'quadrant diagnostics: n, median absolute residual, fraction above 0.03, retained design condition')
    for group in np.unique(quadrant):
        take = quadrant == group
        retained = d[~take]
        print(int(group), int(take.sum()), round(float(np.median(abs(r[take]))),6), round(float(np.mean(abs(r[take])>.03)),3), round(float(np.linalg.cond(np.r_[retained.real, retained.imag])),3))
        worst_survey.append(errors(subset(p, 'survey', ~take), gt))
for label, rows in [('pilot jackknife',worst_pilot), ('survey quadrant deletion',worst_survey)]:
    worst = np.max(rows,axis=0)
    print(label, len(rows), 'worst source / phase / field:', worst, 'max fraction of tolerance:', max(worst/limits))
assert all(np.all(np.max(rows,axis=0) <= limits) for rows in (worst_pilot,worst_survey))
