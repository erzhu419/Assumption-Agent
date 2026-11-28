"""Calibrated quadratic and softplus branches with a two-mode relaxation."""
import numpy as np
from scipy.optimize import minimize_scalar, least_squares


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = map(np.asarray, (panel['u'], panel['v']))
    i, j = np.triu_indices(len(u), 1)
    bias, scale = [], []
    for a, b in zip(u.T, v.T):
        d = a[i] - a[j]
        ok = abs(d) > 1e-8
        slope = np.median((b[i[ok]] - b[j[ok]]) / d[ok])
        r = b - slope * a
        dev = abs(r - np.median(r))
        good = dev <= max(.001, 6 * np.median(dev))
        p, q = _ls(_design(a[good]), b[good])
        bias.append(p)
        scale.append(q)
    return (x - bias) / scale


def _operators(panel, z):
    u = np.asarray(panel['u'])
    def quad(x):
        return _design(x * np.roll(x, -1, axis=1))
    p = quad(z) @ _ls(quad(u), panel['v1'])
    def smooth(x, k):
        return _design(np.logaddexp(0, k*x))
    v = np.asarray(panel['v2'])
    def loss(k):
        d = smooth(u, k)
        return np.sum((d @ _ls(d, v) - v)**2)
    k = minimize_scalar(loss, bounds=(.2, 6.), method='bounded', options={'xatol': 1e-9}).x
    q = smooth(z, k) @ _ls(smooth(u, k), v)
    return p, q


def _features(study, x):
    p, q = _operators(study['panel_c'], _calibrate(study['panel_a'], x))
    diagonal = np.sum(p * q, axis=1)
    endpoint = _design(np.column_stack((p, q, diagonal)))
    fast = np.column_stack((p, diagonal))
    slow = np.column_stack((q, p[:, 0] * q[:, 1]))
    return endpoint, (fast, slow)


def _trajectory(times, traj, blocks, n):
    # Profile out the linear amplitudes for each pair of relaxation rates.
    def design(rates, groups):
        h = -np.expm1(-times[:, None] * rates)
        terms = [g[:, None, :] * h[None, :, j, None] for j, g in enumerate(groups)]
        return np.concatenate(terms, axis=2).reshape(len(groups[0]) * len(times), sum(g.shape[1] for g in groups))

    fit_blocks = tuple(g[:n] for g in blocks)
    obs = traj.ravel()

    def residual(rates):
        d = design(rates, fit_blocks)
        return d @ _ls(d, obs) - obs

    rates = least_squares(residual, [1.82, .56], bounds=([.8, .04], [5., .75])).x
    d = design(rates, fit_blocks)
    pred = design(rates, tuple(g[n:] for g in blocks)) @ _ls(d, obs)
    return pred.reshape(len(blocks[0]) - n, len(times))


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    scalar, blocks = _features(study, np.concatenate((train['x'], test['x'])))
    y = scalar[n:] @ _ls(scalar[:n], train['y'])
    state = _trajectory(np.asarray(study['eval_times']), np.asarray(train['traj']), blocks, n)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
