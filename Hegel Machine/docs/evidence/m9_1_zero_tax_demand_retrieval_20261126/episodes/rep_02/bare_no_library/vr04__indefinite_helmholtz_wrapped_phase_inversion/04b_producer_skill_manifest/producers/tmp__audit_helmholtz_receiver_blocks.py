import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)


def rel(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)


def circular(a, b):
    return abs(np.angle(np.exp(1j*(a-b))))


def errors(result, gt):
    return np.array([rel(result['source_amplitudes'], gt['source_amplitudes']),
                     circular(result['global_phase_rad'], gt['global_phase_rad']),
                     rel(result['predicted_field_abs'], gt['predicted_field_abs'])])


rows = []
for ci, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    i, j = survey['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    quadrant = 2*(i >= nx//2) + (j >= ny//2)
    X = mod._responses(p)[i,j]
    zref = X @ gt['source_amplitudes']
    ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
    order = np.argsort(abs(ref))
    for label, indices in [('weak', order[:len(order)//2]), ('strong', order[len(order)//2:])]:
        sub = {k: v[indices] for k,v in pilot.items()}
        e = errors(mod.invert_helmholtz_sources_and_phase(dict(p, pilot_calibration=sub)), gt)
        print('case', ci, 'pilot half', label, 'source/phase/field', e)
        rows.append(e)
    for q in range(4):
        held = quadrant == q
        sub = {k: v[~held] if k != 'tikhonov_alpha' else v for k,v in survey.items()}
        out = mod.invert_helmholtz_sources_and_phase(dict(p, survey=sub))
        e = errors(out, gt)
        zhat = X[held] @ out['source_amplitudes']
        measured = mod._polar(survey)[held] * np.exp(-1j*out['global_phase_rad'])
        print('case', ci, 'held quadrant', q, 'n', sum(held), 'source/phase/field', e,
              'held complex prediction rel', rel(zhat,zref[held]),
              'held median data residual', np.median(abs(zhat-measured)))
        rows.append(e)
worst = np.max(rows, axis=0)
limits = np.array([.03,.02,.03])
print('blocked audit worst source/phase/field', worst, 'fraction of tolerance', worst/limits)
assert np.all(worst <= limits)
