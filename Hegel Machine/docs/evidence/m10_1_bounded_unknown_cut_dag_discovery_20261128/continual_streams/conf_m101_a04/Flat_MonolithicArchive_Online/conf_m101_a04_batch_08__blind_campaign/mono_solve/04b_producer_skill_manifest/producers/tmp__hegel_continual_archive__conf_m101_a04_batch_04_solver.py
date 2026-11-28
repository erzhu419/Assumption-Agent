"""Recover the calibrated panel-C branches and their scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        a, b = u[keep], v[keep]
        center = a - a.mean(axis=0)
        scale = (center * (b - b.mean(axis=0))).sum(axis=0) / (center**2).sum(axis=0)
        offset = b.mean(axis=0) - a.mean(axis=0) * scale
        error = np.linalg.norm(u * scale + offset - v, axis=1)
        revised = error < max(0.0006, 5 * np.median(error))
        if np.array_equal(keep, revised):
            break
        keep = revised
    return (raw - offset) / scale


def _bilinear(x):
    a, b, c, d = x.T
    return _bias(np.column_stack((a*b, a*d, b*c, c*d)))


def _cosine(x, frequency):
    return _bias(np.column_stack((x, np.cos(frequency * x))))


def _operators(panel, x):
    u = panel['u']
    c = _bilinear(x) @ _fit(_bilinear(u), panel['v1'])
    v = panel['v2']

    def residual(p):
        a = _cosine(u, p[0])
        return (a @ _fit(a, v) - v).ravel()

    frequency = least_squares(residual, [1.5], bounds=(0.15, 8), gtol=1e-12).x[0]
    d = _cosine(x, frequency) @ _fit(_cosine(u, frequency), v)
    return c, d


def _temporal_design(g, h, t, rates):
    rise = -np.expm1(-rates[0] * t)
    tail = t / (1 + rates[1] * t)
    return np.concatenate((g[:, None, :] * rise[None, :, None],
                           h[:, None, :] * tail[None, :, None]), axis=2)


def _fit_trajectory(g, h, t, target):
    width = g.shape[1] + h.shape[1]
    n = len(target)
    values = target.ravel()

    def matrix(rates):
        return _temporal_design(g[:n], h[:n], t, rates).reshape(-1, width)

    def residual(rates):
        a = matrix(rates)
        return a @ _fit(a, values) - values

    rates = least_squares(residual, [1.8, 0.45], bounds=([0.05, 0.001], [6, 2]), gtol=1e-10).x
    return rates, _fit(matrix(rates), values)


def _features(study, raw):
    x = _calibrate(study['panel_a'], raw)
    c, d = _operators(study['panel_c'], x)
    matched = np.sum(c*d, axis=1)
    cross = c[:, 0]*d[:, 1]
    scalar = _bias(np.column_stack((c, d, matched)))
    fast = _bias(np.column_stack((c, matched)))
    slow = _bias(np.column_stack((d, cross)))
    return scalar, fast, slow


def solve_batch(study: dict) -> dict:
    train = study['train']
    t = study['eval_times']
    n = len(train['x'])
    raw = np.concatenate((train['x'], study['test']['x']))
    scalar, fast, slow = _features(study, raw)
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    rates, weights = _fit_trajectory(fast, slow, t, train['traj'])
    state = _temporal_design(fast[n:], slow[n:], t, rates) @ weights
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
