"""Calibrate the panel graph and fit sparse scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root
from sklearn.linear_model import ARDRegression


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    a = _design(u)
    keep = np.argsort(np.linalg.norm(a @ _ls(a, v) - v, axis=1))[:45]
    u, v = u[keep], v[keep]
    cu = np.roll(u, -1, axis=1)
    w = np.column_stack([_ls(_design(u[:, j], cu[:, j], u[:, j] * cu[:, j], u[:, j] ** 3), v[:, j]) for j in range(4)])

    def forward(z):
        q = np.roll(z, -1)
        return w[0] + w[1] * z + w[2] * q + w[3] * z * q + w[4] * z ** 3

    starts = _design(x) @ _ls(_design(v), u)
    return np.array([root(lambda z: forward(z) - row, start, tol=1e-11).x
                     for row, start in zip(x, starts)])


def _odd(z, rate):
    return _design(-np.sign(z) * np.expm1(-rate * np.abs(z)))


def _even(z, rate):
    s = z * z
    return _design(z, s / np.sqrt(1 + rate * s))


def _operator(u, v, f, design, bounds):
    def loss(rate):
        g = design(u, rate)
        return np.linalg.norm(g @ _ls(g, v) - v)

    rate = minimize_scalar(loss, bounds=bounds, method='bounded').x
    return design(f, rate) @ _ls(design(u, rate), v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    b = study['panel_b']
    f = _design(z) @ _ls(_design(b['u']), b['v'])
    p = study['panel_c']
    c = _operator(p['u'], p['v1'], f, _odd, (.9, 2.7))
    d = _operator(p['u'], p['v2'], f, _even, (.1945, .5835))
    z = np.column_stack((f, c, d))
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((z, z[:, i] * z[:, j])), np.column_stack((np.sin(f), np.cos(f), np.sin(c), np.cos(c), np.sin(d), np.cos(d)))


def _time_basis(times, traj):
    def design(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        h = design(rates)
        return (h @ _ls(h, traj.T) - traj.T).ravel()

    return design(least_squares(residual, [1.8, .42], bounds=(.001, 15)).x)


def _ard(a, b, y):
    mean, scale = a.mean(0), np.maximum(a.std(0), 1e-10)
    a, b = (a - mean) / scale, (b - mean) / scale
    return ARDRegression(threshold_lambda=10000).fit(a, y).predict(b)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    g, extra = _features(study, np.vstack((train['x'], study['test']['x'])))
    y = _ard(g[:n], g[n:], train['y'])
    h = _time_basis(study['eval_times'], train['traj'])
    amplitudes = _ls(h, train['traj'].T)
    wide = np.column_stack((g, extra))
    pred = np.column_stack([
        .75 * _ard(wide[:n], wide[n:], a) + .25 * _ard(g[:n], g[n:], a)
        for a in amplitudes
    ]) @ h.T
    terminal = pred[:, -1]
    return {'y_hat': y, 'state_pred': pred,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
