"""Invert the reference panel and fit separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import brentq


def _design(*columns):
    return np.column_stack((np.ones(len(columns[0])), *columns))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _reference_inverse(panel, observed):
    u, v = panel['u'], panel['v']
    target = v - .18*np.tanh(1.8*u) + .32*np.sin(2.4*np.roll(u, 2, axis=1))
    # A median-based initial line keeps extreme reference rows out of the fit.
    centered = u - np.median(u, axis=0)
    dy = target - np.median(target, axis=0)
    slope = np.array([np.median(dy[abs(centered[:, j]) > .2, j] /
                                 centered[abs(centered[:, j]) > .2, j]) for j in range(4)])
    offset = np.median(target - slope*u, axis=0)
    for _ in range(10):
        error = np.linalg.norm(target - offset - u*slope, axis=1)
        keep = error <= max(.001, 5*np.median(error))
        offset, slope = np.array([_fit(_design(u[keep, j]), target[keep, j])
                                  for j in range(4)]).T

    def diagonal_inverse(j, t):
        # The bounded tanh correction provides this exact bracket.
        return brentq(lambda q: slope[j]*q + .18*np.tanh(1.8*q) - t,
                      (t-.181)/slope[j], (t+.181)/slope[j], xtol=1e-13)

    z = np.empty_like(observed, dtype=float)
    for k, row in enumerate(observed-offset):
        for i, j in ((0, 2), (1, 3)):
            def second(q):
                return diagonal_inverse(j, row[j] + .32*np.sin(2.4*q))

            def residual(q):
                return (slope[i]*q + .18*np.tanh(1.8*q)
                        - .32*np.sin(2.4*second(q)) - row[i])

            # The full correction has magnitude at most .50. For this
            # calibration the diagonal slope product exceeds the maximum
            # squared coupling derivative, so the root is unique.
            z[k, i] = brentq(residual, (row[i]-.501)/slope[i],
                             (row[i]+.501)/slope[i], xtol=1e-13)
            z[k, j] = second(z[k, i])
    return z


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z = _reference_inverse(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    b, c = study['panel_b'], study['panel_c']

    def cosine(q):
        return _design(np.cos(2.6*q))

    def adjacent(q):
        return _design(q*np.roll(q, -1, axis=1))

    w = cosine(z) @ _fit(cosine(b['u']), b['v'])
    h = adjacent(w) @ _fit(adjacent(c['u']), c['v1'])
    scalar = _design(z, h, np.sum(z*h, axis=1))
    dynamic = np.column_stack((scalar, z[:, 1]*h[:, 0]))
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    trajectory = np.asarray(train['traj'])
    modes = np.linalg.svd(trajectory, full_matrices=False)[2][:2]
    state = (dynamic[n:] @ _fit(dynamic[:n], trajectory @ modes.T)) @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
