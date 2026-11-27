import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _forward, calibrate_upstream
from batch_solver import solve_batch

J = np.arange(4)
K = (J + 1) % 4

def basis(x, j, kind, a=0.):
    u, v = x[:, j], x[:, K[j]]
    cube, pair = u**3, u*v
    if kind == 'rational':
        den = 1 + a*(u-v)**2
        cube, pair = cube/den, pair/den
    z = np.column_stack((np.ones(len(x)), x, cube, pair))
    if kind == 'difference':
        # Along with uv, these span the sum/difference squares of the pair.
        z = np.column_stack((z, u**2, v**2))
    return z

for item in pickle.load(open('public_batch.pkl', 'rb')):
    ds = item['dataset']
    c = ds['calibration_load_cells']
    x, y = c['true_channel_loads'], c['bridge_raw']
    axis = np.count_nonzero(x, axis=1) == 1
    m = np.flatnonzero(~axis)
    folds = {f'axis-{j}': np.flatnonzero(axis & (x[:, j] != 0)) for j in J}
    folds.update({f'mixed-{i}': r for i, r in enumerate(np.array_split(np.random.default_rng(31).permutation(m), 4))})
    folds['largest-mixed'] = m[np.argsort(np.linalg.norm(x[m], axis=1))[-12:]]
    print('dataset fields:', sorted(ds))
    grid = (0., 1e-4, 1e-3, .01, .1, 1.)
    maxima = []
    for name, idx in folds.items():
        t = np.setdiff1d(np.arange(len(x)), idx)
        bridge = _fit_bridge(x[t], y[t])
        resid = np.linalg.norm(_forward(x[t], bridge)-y[t], axis=1)
        t = t[resid < .01]
        # Score clean held-out observations; the residual gap separates gross spikes.
        idx = idx[np.linalg.norm(_forward(x[idx], bridge)-y[idx], axis=1) < .01]
        scores = [np.linalg.norm(_forward(x[idx], bridge)-y[idx], axis=1).max()]
        params = [('difference', 0.)] + [('rational', a) for a in grid]
        fit_errors, hold_errors = [], []
        for kind, a in params:
            coefs = [np.linalg.lstsq(basis(x[t], j, kind, a), y[t,j], rcond=None)[0] for j in J]
            p = np.column_stack([basis(x[t],j,kind,a)@coefs[j] for j in J])
            q = np.column_stack([basis(x[idx],j,kind,a)@coefs[j] for j in J])
            fit_errors.append(np.linalg.norm(p-y[t]))
            hold_errors.append(np.linalg.norm(q-y[idx], axis=1).max())
        selected = np.argmin(fit_errors[1:])+1
        scores.extend([hold_errors[0], hold_errors[selected]])
        maxima.append(scores)
        print(name, 'cyclic / +squared terms / rational:', np.round(scores, 8), 'train-selected a:', params[selected][1])
    print('worst forward residual per grammar:', np.max(maxima, axis=0))
    coef = _fit_bridge(x, y)
    decoded = calibrate_upstream(ds)
    for name in ('train', 'test', 'chamber_verification_trials'):
        if ds.get(name) is None:
            continue
        for key in ('past_channel_raw', 'hold_channel_raw'):
            z = decoded[name][key].reshape(-1, 4)
            jac = np.broadcast_to(coef[1:5].T, (len(z), 4, 4)).copy()
            jac[:, J, J] += 3*coef[5]*z**2 + coef[6]*z[:, K]
            jac[:, J, K] += coef[6]*z
            s = np.linalg.svd(jac, compute_uv=False)
            err = np.max(np.abs(_forward(z, coef)-ds[name][key].reshape(-1,4)))
            print(name, key, 'minimum singular value:', s[:,-1].min(), 'maximum inverse residual:', err)
            assert s[:,-1].min() > .1 and err < 1e-10
    result = solve_batch(ds)
    assert all(np.isfinite(result[k]).all() for k in ('estimated_taus', 'hereditary_weights', 'predicted_strain', 'initial_rate_pred'))
