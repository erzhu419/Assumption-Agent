import importlib.util
import itertools
import pickle
import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solver = load('/holdout_task/batch_solver.py')
reference = load('/tmp/hegel_continual_archive/beta_m100_c04_batch_06_solver.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
study = item['study']
pulses = item['campaign_diagnostics']
W = np.array([p['w_pulse'] for p in pulses])
Y = np.array([solver._calibrate(dict(study, y_obs_raw=p['y_pulse_raw'])) for p in pulses])
Z = study['Z_nuisance']
Q = np.linalg.qr(Z)[0]
axis_laws = ['periodic', (0, 0), (1, 1), (1, 0), (0, 1)]
laws = list(itertools.product(axis_laws, repeat=2))
errors = []
for law in laws:
    S, _ = reference._forward(study, law)
    d = W @ S.T - Y
    errors.append(np.sum((d - (d @ Q) @ Q.T)**2, axis=1))
errors = np.array(errors)
winner = ('periodic', 'periodic')
for i in range(len(pulses)):
    best = np.argmin(np.delete(errors, i, axis=1).sum(axis=1))
    assert laws[best] == winner, (i, laws[best])
    alternative = np.min(np.delete(errors[:, i], best))**0.5
    print('held-out pulse', i, 'L2', errors[best, i]**0.5, 'closest competing wall L2', alternative)
    assert errors[best, i]**0.5 < min(2e-5, alternative)
S, _ = solver._forward(study)
np.testing.assert_allclose(S, reference._forward(study, winner)[0], rtol=1e-12, atol=1e-12)
d = W @ S.T - Y
print('worst raw tracer RMSE by shift and channel:', np.sqrt(np.mean(d.reshape(len(W), len(study['mu_shifts']), -1, 4)**2, axis=2)).max(axis=0))
cal = study['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
D = np.c_[np.ones(len(x)), x]
pair = np.column_stack([x[:, i]*x[:, j] for i, j in itertools.combinations_with_replacement(range(4), 2)])
for name, design in [('affine', D), ('affine + quadratic interactions', np.c_[D, pair])]:
    e = np.empty_like(y)
    for i in range(5):
        take = np.arange(len(x)) % 5 == i
        coef = np.linalg.lstsq(design[~take], y[~take], rcond=None)[0]
        e[take] = design[take] @ coef - y[take]
    print(name, 'out-of-fold calibration RMSE by channel:', np.sqrt(np.mean(e**2, axis=0)))
