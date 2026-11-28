import contextlib, io, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
sys.path.insert(0, '/tmp')
with contextlib.redirect_stdout(io.StringIO()):
    ctx = runpy.run_path('/tmp/walltest.py')
from downstream_solver import _responses
s, W, Y = (ctx[k] for k in ('s','W','Y'))
mat, kron, eye, splu = (ctx[k] for k in ('mat','kron','eye','splu'))
nx, ny = s['nx'], s['ny']
S, _ = _responses(s)
error = S @ W - Y
n_shift, n_sensor = len(s['mu_shifts']), len(s['sensor_ij'])
err = error.reshape(n_shift, n_sensor, -1)
Z = s['Z_nuisance']
Q = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
models = [('periodic','closed'), ('periodic_no_vel','closed'), ('periodic_plus','closed'), ('closed','closed'), ('open','open'), ('periodic','periodic')]
errs = {}
for x, y in models:
    L = kron(eye(ny), mat('x',x)) + kron(mat('y',y), eye(nx)) + ctx['V']
    pred = np.concatenate([splu((L+t*ctx['E']).tocsc()).solve(ctx['rhs'])[ctx['idx']] for t in s['mu_shifts']])
    errs[x,y] = pred - Y
np.testing.assert_allclose(error, errs[models[0]], atol=1e-12, rtol=0)
shift = np.repeat(np.arange(n_shift), n_sensor)
bank = np.asarray(s['row_bank_id'])
for label, project in [('calibrated', lambda e:e), ('nuisance-filtered', lambda e:Q@e)]:
    candidates = {k:project(e) for k,e in errs.items()}
    margins = []
    if label == 'calibrated':
        for h in range(n_shift):
            train = shift != h
            for p in range(W.shape[1]):
                keep = np.arange(W.shape[1]) != p
                scores = sorted((np.sqrt(np.mean(e[train][:,keep]**2)),k) for k,e in candidates.items())
                assert scores[0][1] == models[0], scores
                held = np.sqrt(np.mean(candidates[scores[0][1]][~train,p]**2))
                assert held < 3e-6, held
                margins.append(scores[1][0]/scores[0][0])
        print('Pulse-and-shift exclusion folds:', len(margins), 'minimum training RMSE runner-up ratio:', min(margins))
    ratios, top = [], []
    for h in range(n_shift):
        for b in np.unique(bank):
            mask = (shift==h)&(bank==b)
            for p in range(W.shape[1]):
                scores = sorted((np.sqrt(np.mean(e[mask,p]**2)),k) for k,e in candidates.items())
                assert scores[0][1] == models[0], (label,h,b,p,scores)
                top.append(scores[0][0]); ratios.append(scores[1][0]/scores[0][0])
    assert max(top) < 3e-6, max(top)
    print(label, 'pulse x shift x bank slices:', len(top), 'worst RMSE:', max(top), 'minimum runner-up ratio:', min(ratios))
