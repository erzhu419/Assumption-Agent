import pickle
import sys
from unittest.mock import patch

import numpy as np
from scipy.sparse import diags

sys.path.insert(0, '/holdout_task')
import downstream_solver as solver
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    items = pickle.load(stream)
periodic = solver._axis

def reflecting(kappa, velocity, h):
    n = len(kappa)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h, -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2, -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[-upper[0], -lower[:-1] - upper[1:], -lower[-1]]
    return diags((lower, main, upper), (-1, 0, 1), shape=(n, n), format='csc')

errors = {name: [] for name in ('periodic_xy', 'reflect_x', 'reflect_y', 'reflect_xy')}
for item in items:
    study = item['study']
    pulses = item.get('campaign_diagnostics') or []
    if not pulses:
        continue
    observed = np.stack([calibrate_upstream({**study, 'y_obs_raw': pulse['y_pulse_raw']}, None)['y_obs_raw'] for pulse in pulses])
    weights = np.stack([pulse['w_pulse'] for pulse in pulses])
    for name, reflected in (('periodic_xy', set()), ('reflect_x', {0}), ('reflect_y', {1}), ('reflect_xy', {0, 1})):
        axes = iter(range(2))
        def axis(kappa, velocity, h):
            law = reflecting if next(axes) in reflected else periodic
            return law(kappa, velocity, h)
        with patch.object(solver, '_axis', axis):
            response, _ = solver._forward(study)
        residual = (weights @ response.T - observed).reshape(len(pulses), len(study['mu_shifts']), -1)
        errors[name].extend(np.sqrt(np.mean(residual**2, axis=-1)).ravel())

reference = np.asarray(errors['periodic_xy'])
assert reference.size > 0
for name, values in errors.items():
    values = np.asarray(values)
    print(f'{name}: slices={values.size}, worst_RMSE={values.max():.6e}, best_RMSE={values.min():.6e}, slices_worse_than_periodic={np.count_nonzero(values > reference)}')
assert reference.max() < 2e-6
for name in ('reflect_x', 'reflect_y'):
    assert max(errors[name]) > 10 * reference.max(), f'{name}: the seam did not clear the ablation margin'
