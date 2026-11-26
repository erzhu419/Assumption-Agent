import pickle
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
import helmholtz_phase_inversion as h

with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

worst = np.zeros(3)
for ci, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    s = p['survey']
    ij = s['sensor_ij']
    shape = np.shape(p['n_sq'])
    i, j = ij.T
    depth = np.minimum.reduce([i, j, shape[0]-1-i, shape[1]-1-j])
    blocks = [(f'quadrant_{a}{b}', ((i >= shape[0]//2) == a) & ((j >= shape[1]//2) == b))
              for a in (0, 1) for b in (0, 1)]
    blocks.append(('near_wall', depth <= 2))
    responses = h._responses(p)
    design = responses[np.ravel_multi_index(ij.T, shape)]
    y = h._phasors(s) * np.exp(-1j*truth['global_phase_rad'])
    reference = design @ truth['source_amplitudes']
    residual = np.abs(y-reference)
    pilot = p['pilot_calibration']
    known = pilot['known_field_re'] + 1j*pilot['known_field_im']
    unit_phase = np.angle(np.sum(h._phasors(pilot) * known.conj() / (pilot['amplitude_obs'] * abs(known))))
    base_phase = h.invert_helmholtz_sources_and_phase(p)['global_phase_rad']
    circ = lambda x: abs(np.angle(np.exp(1j*(x-truth['global_phase_rad']))))
    ls = np.linalg.lstsq(np.vstack([design.real, design.imag]), np.r_[y.real, y.imag], rcond=None)[0]
    print('case', ci, 'plain least squares source error', np.linalg.norm(ls-truth['source_amplitudes'])/np.linalg.norm(truth['source_amplitudes']), 'pilot weighted/unit phase errors', circ(base_phase), circ(unit_phase))
    for name, held in blocks:
        reduced = dict(s)
        for key in ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs'):
            reduced[key] = s[key][~held]
        result = h.invert_helmholtz_sources_and_phase(dict(p, survey=reduced))
        e = np.array([np.linalg.norm(result[k]-truth[k])/np.linalg.norm(truth[k]) for k in ('source_amplitudes', 'predicted_field_abs')] + [circ(result['global_phase_rad'])])
        worst = np.maximum(worst, e)
        pred = design[held] @ result['source_amplitudes']
        print(name, 'held', held.sum(), 'source/field/phase', e, 'held median clean-field discrepancy', np.median(np.abs(pred-reference[held])), 'held median observed discrepancy', np.median(residual[held]))
print('worst spatial-block source/field/phase', worst)
assert np.all(worst <= [.030, .030, .020]), worst
