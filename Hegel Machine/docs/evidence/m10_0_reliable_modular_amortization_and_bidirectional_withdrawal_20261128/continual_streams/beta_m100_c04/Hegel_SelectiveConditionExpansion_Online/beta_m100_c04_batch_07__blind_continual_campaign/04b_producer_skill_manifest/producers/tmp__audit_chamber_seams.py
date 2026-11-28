import itertools
import sys

import numpy as np

sys.path[:0] = ['/tmp', '/holdout_task']
from fit_wall import axes, predict, s, W, Y, payload
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward

key = ('periodic', 'periodic')
models = {k: predict(*axes(*k)) for k in itertools.product(('robin', 'reflect', 'periodic'), repeat=2)}
cal = calibrate_upstream(s, payload)
S, _ = _forward(cal)
np.testing.assert_allclose(S @ W, models[key], atol=1e-13, rtol=0)
for name, a in zip('xy', axes(*key)):
    np.testing.assert_allclose(a @ np.ones(a.shape[0]), 0, atol=1e-11, rtol=0)
    assert a[0,-1] < 0 and a[-1,0] < 0, name

# Hold out each spatial rosette at every shift and for every pulse. Refit
# each candidate's nuisance profile on the remaining rows before selection.
Z = cal['Z_nuisance']
nshift = len(s['mu_shifts'])
nsensor = len(s['sensor_ij'])
ros = np.tile(np.arange(nsensor) // 4, nshift)
worst = 0.
smallest_margin = np.inf
for r in np.unique(ros):
    train = ros != r
    test = ~train
    scores = {}
    for k, predicted in models.items():
        err = predicted[train] - Y[train]
        z = Z[train]
        beta = np.linalg.solve(z.T @ z + cal['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ err)
        scores[k] = np.sqrt(np.mean((err - z @ beta)**2))
    ordered = sorted(scores, key=scores.get)
    assert ordered[0] == key, (r, scores)
    smallest_margin = min(smallest_margin, scores[ordered[1]] - scores[key])
    residual = ((S @ W)[test] - Y[test]).reshape(nshift, 4, W.shape[1])
    worst = max(worst, np.sqrt(np.max(np.mean(residual**2, axis=1))))
print(f'Periodic/periodic selected in {len(np.unique(ros))} rosette holdouts; smallest nuisance-profiled training RMSE lead: {smallest_margin:.6g}')
print(f'Delivered operator worst held-out rosette/shift/pulse RMSE: {worst:.6g}')
assert worst < 1e-5, worst
