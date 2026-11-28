"""Predict the response and trajectory with separate kernel models."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _rbf(x, xt, length):
    return np.exp(-cdist(x, xt, 'sqeuclidean') / (2 * length**2))


def _predict(k, cross, h, ht, target, nugget):
    k = k.copy()
    k.flat[::len(k) + 1] += nugget
    n = h.shape[1]
    w = cho_solve(cho_factor(k), np.column_stack((h, target)))
    a, b = w[:, :n], w[:, n:]
    beta = np.linalg.solve(h.T @ a, h.T @ b)
    return ht @ beta + cross @ (b - a @ beta)


def _response(x, xt, y, h, ht):
    scale = x.std(axis=0)
    q, qt = x / scale, xt / scale
    return _predict(_rbf(q, q, 2), _rbf(qt, q, 2), h, ht, y[:, None], .001)[:, 0]


def _trajectory(study, x, xt, z, h, ht):
    a = study['panel_a']
    # Coordinatewise median slopes/intercepts reject sparse calibration outliers.
    scale, bias = [], []
    for u, v in zip(a['u'].T, a['v'].T):
        i, j = np.triu_indices(len(u), 1)
        dx = u[i] - u[j]
        keep = np.abs(dx) > 1e-9
        slope = np.median((v[i] - v[j])[keep] / dx[keep])
        scale.append(slope)
        bias.append(np.median(v - slope * u))
    b = study['panel_b']
    coef = np.linalg.lstsq(_linear(b['u']), b['v'], rcond=None)[0]
    q = _linear((x - bias) / scale) @ coef
    qt = _linear((xt - bias) / scale) @ coef
    def features(u):
        return np.column_stack((u, np.tanh(.7 * u)))
    f, ft = features(q), features(qt)
    mu, sd = f.mean(axis=0), f.std(axis=0)
    f, ft = (f - mu) / sd, (ft - mu) / sd
    def kernel(p, r, u, v):
        return .25 * (1 + p @ r.T / f.shape[1])**2 + .2 * _rbf(u, v, 1.35)
    basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    amplitudes = _predict(kernel(f, f, x, x), kernel(ft, f, xt, x), h, ht, z @ basis.T, .001)
    return amplitudes @ basis


def solve_batch(study):
    train = study['train']
    x, xt = train['x'], study['test']['x']
    h, ht = _linear(x), _linear(xt)
    y = _response(x, xt, train['y'], h, ht)
    state = _trajectory(study, x, xt, train['traj'], h, ht)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
