"""Robust coordinate calibration, panel-B operator, and separate readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    gain, offset = np.empty(u.shape[1]), np.empty(u.shape[1])
    i, j = np.triu_indices(len(u), 1)
    for k in range(u.shape[1]):
        a, b = u[:, k], v[:, k]
        du = a[i] - a[j]
        good = np.abs(du) > 1e-8
        g = np.median((b[i[good]] - b[j[good]]) / du[good])
        c = np.median(b - g * a)
        r = np.abs(b - g * a - c)
        keep = r <= max(5 * np.median(r), 1e-8)
        if np.count_nonzero(keep) < 3:
            keep = np.argsort(r)[:max(3, len(a) // 2)]
        gain[k], offset[k] = _ls(np.c_[a[keep], np.ones(np.count_nonzero(keep))], b[keep])
    return gain, offset


def _b_features(q, p):
    scale, power, exponent = p
    logabs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0, power * (np.log(scale) + logabs)))
    return np.c_[np.ones(len(q)), q, h]


def _fit_b(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    def residual(p):
        f = _b_features(u, p)
        return (f @ _ls(f, v) - v).ravel()
    p = least_squares(residual, (2.5, 12., .036), bounds=((.3, 2., .0001), (6., 30., .5)), max_nfev=200).x
    return p, _ls(_b_features(u, p), v)


def _readout(b):
    a = np.c_[np.ones(len(b)), b]
    return a, np.c_[a, b[:, 0] * b[:, 2], b[:, 1] * b[:, 3]]


def solve_batch(study: dict) -> dict:
    train = study['train']
    gain, offset = _calibrate(study['panel_a'])
    p, w = _fit_b(study['panel_b'])
    def features(x):
        return _readout(_b_features((np.asarray(x) - offset) / gain, p) @ w)
    a, f = features(train['x'])
    a_new, f_new = features(study['test']['x'])
    y = f_new @ _ls(f, train['y'])
    t = a_new @ _ls(a, train['traj'])
    last = t[:, -1]
    return {'y_hat': y, 'state_pred': t,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
