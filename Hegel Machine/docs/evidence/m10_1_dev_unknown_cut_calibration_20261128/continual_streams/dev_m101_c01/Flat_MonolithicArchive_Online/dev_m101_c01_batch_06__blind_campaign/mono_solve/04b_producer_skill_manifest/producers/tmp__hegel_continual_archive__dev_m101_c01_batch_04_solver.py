from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _quadratic(x):
    return np.column_stack([np.ones(len(x)), x, *(x[:, i] * x[:, j]
        for i, j in combinations_with_replacement(range(x.shape[1]), 2))])


def _readout(x, xt, y):
    p, pt = _quadratic(x), _quadratic(xt)
    a = np.column_stack([np.ones(len(x)), x])
    at = np.column_stack([np.ones(len(xt)), xt])
    d = np.sum((x[:, None] - x[None, :])**2, axis=2)
    dt = np.sum((xt[:, None] - x[None, :])**2, axis=2)
    k, kt = p @ p.T, pt @ p.T
    pred = np.zeros(len(xt))
    for weight, length, scale, noise, dot in (
        (0.65, 1.5, 10., .002, True),
        (0.35, 1.2, 10., .005, False),
    ):
        g = np.exp(-d / (2 * length**2))
        gt = np.exp(-dt / (2 * length**2))
        if dot:
            g *= a @ a.T
            gt *= at @ a.T
        factor = cho_factor(k + scale * g + noise * np.eye(len(x)))
        pred += weight * ((kt + scale * gt) @ cho_solve(factor, y))
    return pred


def _state(x, xt, z, panel_a, panel_b):
    # Panel A maps reference coordinates to measured coordinates. Reject
    # gross calibration outliers before estimating the diagonal map.
    u, v = panel_a['u'], panel_a['v']
    good = np.max(np.abs(v), axis=1) < 2
    a = np.linalg.lstsq(np.column_stack((np.ones(good.sum()), u[good])),
                        v[good], rcond=1e-10)[0]
    u, v = panel_b['u'], panel_b['v']
    b = np.linalg.lstsq(np.column_stack((np.ones(len(u)), u)), v, rcond=1e-10)[0]
    transform = b[1:] / a[1:].diagonal()[:, None]
    w, wt = x @ transform, xt @ transform
    mode = np.linalg.svd(z, full_matrices=False)[2][0]
    amplitude = z @ mode
    p, pt = _quadratic(x), _quadratic(xt)
    d = (w[:, None] - w[None, :])**2
    dt = (wt[:, None] - w[None, :])**2
    k, kt = p @ p.T, pt @ p.T
    eye = np.eye(len(x))

    def kernel(d, base, h):
        r = np.sqrt(5 * (d @ np.exp(h[:4])))
        return base + np.exp(h[4]) * (1 + r + r*r/3) * np.exp(-r)

    def objective(h):
        factor = cho_factor(kernel(d, k, h) + 0.00004 * eye)
        alpha = cho_solve(factor, amplitude)
        return (0.5 * amplitude @ alpha + np.log(factor[0].diagonal()).sum()
                + .8 * np.sum((h[:4] - h[:4].mean())**2))

    h = minimize(objective, [-1.5]*4 + [0.],
                 bounds=[(-6, 2)]*4 + [(-8, 5)],
                 method='L-BFGS-B', options={'maxiter': 45}).x
    factor = cho_factor(kernel(d, k, h) + 0.00004 * eye)
    amp_test = kernel(dt, kt, h) @ cho_solve(factor, amplitude)
    return amp_test[:, None] * mode


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = np.asarray(train['x']), np.asarray(study['test']['x'])
    y = _readout(x, xt, np.asarray(train['y']))
    state = _state(x, xt, np.asarray(train['traj']), study['panel_a'], study['panel_b'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
