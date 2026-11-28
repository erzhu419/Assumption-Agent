"""Invert the reference calibration and fit separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, obs, upstream):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    gain, offset = upstream['gain'], upstream['offset']
    delta = np.max(np.abs(v - u * gain - offset), axis=1)
    m = np.median(delta)
    good = delta < m + max(.42, 5 * np.median(abs(delta - m)))
    def design(q, p, j):
        a, b = q[:, j], q[:, (j + 2) % 4]
        return np.c_[np.ones(len(q)), a, b, np.sin(p[0]*a), np.sin(p[1]*b), a**3]
    def fit(p, mask):
        ds = [design(u[mask], p, j) for j in range(4)]
        co = [_ls(a, v[mask, j]) for j, a in enumerate(ds)]
        return co, np.column_stack([a @ w - v[mask, j] for j, (a,w) in enumerate(zip(ds,co))])
    def opt(mask):
        return least_squares(lambda p: fit(p, mask)[1].ravel(), [2.4, 2.4], bounds=([1., 1.], [4., 4.])).x
    p = opt(good)
    c, _ = fit(p, good)
    def forward(q):
        return np.column_stack([design(q, p, j) @ c[j] for j in range(4)])
    err = np.max(abs(forward(u) - v), axis=1)
    med = np.median(err)
    good = err < max(.015, 6 * med)
    if good.sum() >= 20:
        p = opt(good)
        c, _ = fit(p, good)
    def inv(row):
        init = (row - offset) / gain
        fun = lambda x: forward(x[None,:])[0] - row
        sol = root(fun, init)
        if np.linalg.norm(fun(sol.x)) > 1e-7:
            sol = least_squares(fun, np.clip(init, -1.49, 1.49), bounds=(-1.5, 1.5))
        return sol.x
    return np.array([inv(row) for row in obs])


def _mapped(q, fitted):
    scale, power, exponent = fitted['p']
    logabs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0, power * (np.log(scale) + logabs)))
    return np.c_[np.ones(len(q)), q, h] @ fitted['w']


def _scalar(q, b, c, y, k):
    h = q[:, 2]
    gate = 1 / (1+k*h*h)
    bg = b*gate[:, None]
    design = np.c_[np.ones(len(q)), bg, c, bg*h[:,None]**3]
    return design[len(y):] @ _ls(design[:len(y)], y)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    fitted = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    q = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])), fitted)
    b = _mapped(q, fitted)
    c = b*np.roll(q, -1, axis=1)
    y = _scalar(q, b, c, np.asarray(train['y']), float(study['telemetry_vec'][2]))
    design = np.c_[np.ones(len(q)), b, c]
    state = design[n:] @ _ls(design[:n], train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
