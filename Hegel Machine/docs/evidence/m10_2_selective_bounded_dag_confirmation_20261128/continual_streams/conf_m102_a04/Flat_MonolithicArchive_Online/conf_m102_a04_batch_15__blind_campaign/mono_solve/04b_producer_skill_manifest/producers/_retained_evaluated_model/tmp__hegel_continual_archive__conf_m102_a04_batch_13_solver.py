"""Recover calibrated panel operators and predict a two-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ols(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _calibrate(panel, raw):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    offset, scale = [], []
    row1, row2 = np.triu_indices(len(u), 1)
    for a, b in zip(u.T, v.T):
        delta = a[row1] - a[row2]
        keep = np.abs(delta) > 1e-8
        slope = np.median((b[row1[keep]] - b[row2[keep]]) / delta[keep])
        center = np.median(b - slope * a)
        f = _intercept(a)
        for _ in range(5):
            r = np.abs(b - center - slope * a)
            keep = r <= max(5 * np.median(r), 0.00035)
            center, slope = _ols(f[keep], b[keep])
        offset.append(center)
        scale.append(slope)
    return (raw - np.asarray(offset)) / np.asarray(scale)


def _profile(design, target, start, bounds):
    def residual(q):
        f = design(q)
        return (f @ _ols(f, target) - target).ravel()

    q = least_squares(residual, start, bounds=bounds, gtol=1e-12).x
    return q, _ols(design(q), target)


def _operators(panel, z):
    u = np.asarray(panel['u'])
    i, j = np.triu_indices(u.shape[1], 1)

    def signed(x, q):
        return _intercept(x / (1 + q[0] * np.abs(x)) + q[1] * x)

    def paired(x, q):
        return _intercept(np.column_stack((np.tanh(q[0] * x), x[:, i] * x[:, j])))

    a, wa = _profile(lambda q: signed(u, q), panel['v1'], [.83, -.16], ([.3, -.4], [2., .4]))
    b, wb = _profile(lambda q: paired(u, q), panel['v2'], [1.05], (.4, 2.))
    return signed(z, a) @ wa, paired(z, b) @ wb


def _readout(features, target, n, regularization):
    f = (features - features[:n].mean(axis=0)) / np.maximum(features[:n].std(axis=0), 1e-6)
    f = _intercept(f)
    penalty = np.eye(f.shape[1]) * np.sqrt(regularization)
    penalty[0, 0] = 0
    target = np.asarray(target)
    zeros = np.zeros((len(penalty),) + target.shape[1:])
    w = _ols(np.concatenate((f[:n], penalty)), np.concatenate((target, zeros)))
    return f[n:] @ w


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    raw = np.concatenate((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], raw)
    g, h = _operators(study['panel_c'], z)
    y = _readout(np.column_stack((g, h, np.sum(g * h, axis=1))), train['y'], n, .0001)
    t = np.asarray(study['eval_times'])

    def basis(q):
        return -np.expm1(-t[:, None] * q)

    rates, amplitudes = _profile(basis, train['traj'].T, [.4, 1.8], ([.05, .8], [.8, 4.]))
    cross = (g[:, :, None] * h[:, None, :]).reshape(len(z), -1)
    state = _readout(np.column_stack((g, h, cross)), amplitudes.T, n, .0001) @ basis(rates).T
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
