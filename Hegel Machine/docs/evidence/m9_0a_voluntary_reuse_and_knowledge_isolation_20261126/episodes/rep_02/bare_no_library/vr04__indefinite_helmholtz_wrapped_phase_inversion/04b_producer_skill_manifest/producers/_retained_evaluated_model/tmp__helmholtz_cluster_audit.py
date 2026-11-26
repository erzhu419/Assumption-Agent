import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

scores = []
for case in cases:
    p, gt = case['problem'], case['gt']
    sv = p['survey']
    ij = np.asarray(sv['sensor_ij'])
    phase = gt['global_phase_rad']
    rot = np.exp(1j * phase)
    a, u = gt['source_amplitudes'], gt['predicted_field_abs']
    clean = s._response(p)[ij[:, 0], ij[:, 1]] @ a
    residual = s._polar(sv) / rot - clean
    n = len(ij)
    count = round(.07 * n)
    order = np.argsort(np.abs(residual))
    spike_idx = order[-count:]
    inlier_idx = order[:-count]
    for anchor in np.unique(ij, axis=0):
        near = np.argsort(np.sum((ij - anchor)**2, axis=1), kind='stable')[:count]
        # Swap measured residuals to cluster the spikes without altering their distribution.
        moved = residual.copy()
        src = np.setdiff1d(spike_idx, near)
        dst = np.intersect1d(near, inlier_idx)
        moved[dst], moved[src] = moved[src], moved[dst]
        z = rot * (clean + moved)
        out = s.invert_helmholtz_sources_and_phase(dict(p, survey=dict(sv, amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))))
        scores.append((np.linalg.norm(out['source_amplitudes'] - a) / np.linalg.norm(a),
                       abs(np.angle(np.exp(1j * (out['global_phase_rad'] - phase)))),
                       np.linalg.norm(out['predicted_field_abs'] - u) / np.linalg.norm(u)))
worst = np.max(scores, axis=0)
print('Measured-spike spatial clusters:', len(scores), 'worst source/phase/field:', worst, flush=True)
assert np.all(worst <= [.03, .02, .03]), worst
