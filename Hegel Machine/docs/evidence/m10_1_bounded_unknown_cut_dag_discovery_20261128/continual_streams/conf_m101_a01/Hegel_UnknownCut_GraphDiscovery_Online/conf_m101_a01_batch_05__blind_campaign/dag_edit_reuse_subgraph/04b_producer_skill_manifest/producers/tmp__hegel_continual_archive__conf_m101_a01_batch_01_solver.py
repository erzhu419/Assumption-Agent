from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _kernel_fit(x, xt, trend, trendt, y, length, nugget):
    factor = 2 * x.shape[1] * length**2
    k = np.exp(-cdist(x, x, 'sqeuclidean') / factor)
    k.flat[::len(x) + 1] += nugget
    solved = cho_solve(cho_factor(k), np.column_stack((trend, y)))
    w, v = solved[:, :-1], solved[:, -1]
    beta = np.linalg.solve(trend.T @ w, trend.T @ v)
    cross = np.exp(-cdist(xt, x, 'sqeuclidean') / factor)
    return trendt @ beta + cross @ (v - w @ beta)


def _features(x, xt, y, trajectories):
    scale = x.std(axis=0)
    x, xt = x / scale, xt / scale
    d, dt = _linear(x), _linear(xt)
    _, _, vt = np.linalg.svd(trajectories, full_matrices=False)
    profile = vt[0]
    z = trajectories @ profile
    zp = _kernel_fit(x, xt, d, dt, z, 1.7, 1e-4)
    pairs = list(combinations_with_replacement(range(x.shape[1]), 2))
    quadratic = np.column_stack([d] + [x[:, i] * x[:, j] for i, j in pairs])
    coef = np.linalg.lstsq(quadratic, y, rcond=None)[0][d.shape[1]:]
    h = np.zeros((x.shape[1], x.shape[1]))
    for (i, j), c in zip(pairs, coef):
        h[i, j] = h[j, i] = c / (1 + (i != j))
    val, vec = np.linalg.eigh(h)
    importance = abs(val) / max(abs(val))
    return x, xt, z / z.std(), zp / z.std(), importance, vec, zp[:, None] * profile


def _predict(features, y, cfg):
    x, xt, z, zp, imp, vec, _ = features
    mix, stretch, warp, latent_metric, use_latent, length, nugget = cfg
    d, dt = _linear(x), _linear(xt)
    if use_latent:
        d, dt = np.column_stack((d, z)), np.column_stack((dt, zp))
    if warp:
        x, xt = x + warp * x*x, xt + warp * xt*xt
    if mix:
        weights = np.sqrt((1 - mix) + mix * (0.6 + stretch * imp))
        x, xt = (x @ vec) * weights, (xt @ vec) * weights
    if latent_metric:
        # Keep the original kernel's distance normalization after adding a coordinate.
        scale = np.sqrt((x.shape[1] + 1) / x.shape[1])
        x = np.column_stack((x, latent_metric * z)) * scale
        xt = np.column_stack((xt, latent_metric * zp)) * scale
    return _kernel_fit(x, xt, d, dt, y, length, nugget)


def solve_batch(study):
    train = study['train']
    y = train['y']
    features = _features(train['x'], study['test']['x'], y, train['traj'])
    state = features[-1]
    pred = sum(w * _predict(features, y, cfg) for w, cfg in _SCALAR_MODELS)
    last = state[:, -1]
    return {'y_hat': pred, 'state_pred': state,
            'diagnostics_vec': np.array([pred.mean(), pred.std(), last.mean(), last.std()])}


_SCALAR_MODELS = [(0.375, (1.0, 0.4, 0.0, 0.55, 0, 1.12, 3e-06)), (0.375, (1.0, 1.5, -0.075, 0.0, 1, 0.82, 3e-06)), (0.25, (1.0, 0.4, 0.075, 0.55, 0, 1.34, 3e-06))]
