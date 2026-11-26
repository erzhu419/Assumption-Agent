import pickle
import runpy
import numpy as np

solver = runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)

def circular(a):
    return abs(np.angle(np.exp(1j*a)))

worst = np.zeros(3)
worst_drift = 0.
for case in cases:
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    known = pilot['known_field_re'] + 1j*pilot['known_field_im']
    drop = np.argmax(abs(known))
    reduced_pilot = {k: np.delete(v, drop) for k, v in pilot.items()}
    i, j = survey['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    quadrant = 2*(i >= nx/2).astype(int) + (j >= ny/2).astype(int)
    a = solver['_responses'](p)[i,j]
    raw = solver['_phasors'](survey)
    for g in np.unique(quadrant):
        held = quadrant == g
        subset = {**survey, **{k: survey[k][~held] for k in ('sensor_ij','amplitude_obs','wrapped_phase_obs')}}
        result = solver['invert_helmholtz_sources_and_phase']({**p, 'pilot_calibration': reduced_pilot, 'survey': subset})
        x, phase = result['source_amplitudes'], result['global_phase_rad']
        err = [rel(x,gt['source_amplitudes']), rel(result['predicted_field_abs'],gt['predicted_field_abs']), circular(phase-gt['global_phase_rad'])]
        worst = np.maximum(worst, err)
        pred = a @ x
        aligned = raw * np.exp(-1j*phase)
        residual = abs(aligned-pred)
        # Fit and choose the impulse cutoff on training receivers only.
        clean_held = held & (residual <= 3.5*np.median(residual[~held]))
        drift = abs(np.angle(np.vdot(pred[clean_held], aligned[clean_held])))
        worst_drift = max(worst_drift, drift)
        print(f'quadrant {g}: {clean_held.sum()}/{held.sum()} retained, held-out phase discrepancy {drift:.6f} rad, errors {err}')
print('Worst joint holdouts (source, field, phase):', worst)
print('Worst held-out survey-to-pilot phase discrepancy:', worst_drift)
assert np.all(worst <= [.030,.030,.020])
assert worst_drift <= .020, 'Inspect coherent spatial phase residual before accepting a shared oscillator phase'
