"""Robust calibration, thresholded panel operator, and paired-index readout."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    bias, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    for k, (a, b) in enumerate(zip(u.T, v.T)):
        dx = a[j] - a[i]
        good = np.abs(dx) > 0.1 * np.ptp(a)
        slope = np.median((b[j[good]] - b[i[good]]) / dx[good])
        r = b - slope * a
        med = np.median(r)
        keep = np.abs(r - med) <= max(1e-6, 8*np.median(np.abs(r-med)))
        bias[k], gain[k] = _fit(_affine(a[keep]), b[keep])
    return bias, gain


def _panel_operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def features(a, p):
        threshold, power = p
        return _affine(np.sign(a) * np.maximum(np.abs(a) - threshold, 0)**power)

    def residual(p):
        h = features(u, p)
        return (h @ _fit(h, v) - v).ravel()

    p = least_squares(residual, [.22, 1.25], bounds=([0, .1], [.7, 8]),
                      gtol=1e-12, max_nfev=300).x
    return features(z, p) @ _fit(features(u, p), v)


def _readout(d):
    return np.column_stack((_affine(d), d[:, 0]*d[:, 2] + d[:, 1]*d[:, 3]))


def _link(index, k):
    return index / (1 + k*np.abs(index))


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x, xt = (np.asarray(p['x'], dtype=float) for p in (tr, study['test']))
    bias, gain = _calibrate(study['panel_a'])
    d = _panel_operator(study['panel_b'], (np.concatenate((x, xt)) - bias)/gain)
    h = _readout(d)
    ht, he = h[:len(x)], h[len(x):]
    y = np.asarray(tr['y'], dtype=float)
    p0 = np.r_[_fit(ht, y), .22]
    lim = np.full(ht.shape[1], np.inf)
    opt = least_squares(lambda p: _link(ht @ p[:-1], p[-1]) - y, p0,
                        bounds=(np.r_[-lim, 1e-5], np.r_[lim, 8]), gtol=1e-11)
    y_hat = _link(he @ opt.x[:-1], opt.x[-1])
    state_pred = he[:, :-1] @ _fit(ht[:, :-1], np.asarray(tr['traj'], dtype=float))
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
