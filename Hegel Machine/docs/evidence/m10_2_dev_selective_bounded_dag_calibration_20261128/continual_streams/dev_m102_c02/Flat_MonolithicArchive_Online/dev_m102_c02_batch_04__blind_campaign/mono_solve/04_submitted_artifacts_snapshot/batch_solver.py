"""Robust cyclic calibration, panel-B operator, and shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    r = np.roll(u, -1, axis=1)
    d = u.shape[1]
    j = np.arange(d)
    h = np.zeros((len(u), d, 2*d+3))
    h[:, j, j] = 1
    h[:, j, d+j] = u
    h[:, :, -3:] = np.stack((r, u*r, u**3), axis=-1)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(12):
        p = _fit(h[keep].reshape(-1, h.shape[-1]), v[keep].ravel())
        error = np.linalg.norm(h @ p-v, axis=1)
        new = error <= max(5*np.median(error), 1e-8)
        if np.array_equal(keep, new) or new.sum() < 2*d+3:
            break
        keep = new
    bias, gain = p[:d], p[d:2*d]
    linear, pair, cubic = p[-3:]

    def forward(z):
        other = np.roll(z, -1)
        return bias + gain*z + linear*other + pair*z*other + cubic*z**3

    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/gain,
                                   gtol=1e-11).x for row in x])


def _operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def transform(x, p):
        return _affine(x + p[0]*x/(1+p[1]*x*x))

    def res(p):
        h = transform(u, p)
        return (h @ _fit(h, v)-v).ravel()

    p = least_squares(res, [-1, .8], bounds=([-4,.001],[4,12]),
                      max_nfev=150).x
    return transform(z, p) @ _fit(transform(u, p), v)


def _state(h, yt, q, t):
    coeff = _fit(h, yt)

    def project(log_rates):
        modes = -np.expm1(-np.exp(log_rates[:, None])*t)
        return _fit(modes.T, coeff.T).T @ modes

    opt = least_squares(lambda p: (h @ (project(p)-coeff)).ravel(),
                        np.log([.4, 1.8]), bounds=(np.log([.005,.25]), np.log([2,15])))
    return q @ project(opt.x)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    z = _calibration(study['panel_a'], np.concatenate((train['x'],test['x'])))
    w = _operator(study['panel_b'], z)
    paired = np.sum(w[:, :2]*w[:, 2:], axis=1)
    h = np.column_stack((_affine(w), paired))
    y = h[n:] @ _fit(h[:n], train['y'])
    state = _state(h[:n], train['traj'], h[n:], np.asarray(study['eval_times'], dtype=float))
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}
