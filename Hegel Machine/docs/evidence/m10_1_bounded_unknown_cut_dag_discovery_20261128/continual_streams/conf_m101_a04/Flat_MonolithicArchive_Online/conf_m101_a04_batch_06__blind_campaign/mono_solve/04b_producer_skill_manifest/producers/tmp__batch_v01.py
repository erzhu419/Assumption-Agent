"""Calibrate the inputs and predict both outputs with a shared residual kernel."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_basis(x):
    return np.stack((np.ones_like(x), x, x**3,
                     (x-1)*np.roll(x, -1, axis=1)), axis=2)


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(v, axis=1) < 5
    a = _calibration_basis(u[keep])
    w = np.stack([_fit(a[:, i], v[keep, i]) for i in range(4)])
    return np.stack([least_squares(lambda x: (_calibration_basis(x[None])*w).sum(2)[0]-r,
                                   r, gtol=1e-11).x for r in raw])


def _bilinear(x):
    a, b, c, d = x.T
    return _bias(np.column_stack((a*b, a*d, b*c, c*d)))


def _branches(panel, x):
    u = panel['u']
    c = _bilinear(x) @ _fit(_bilinear(u), panel['v1'])
    d = _bias(np.cos(2.4*x)) @ _fit(_bias(np.cos(2.4*u)), panel['v2'])
    return c, d


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    raw = np.concatenate((train['x'], study['test']['x']))
    x = _calibrate(study['panel_a'], raw)
    c, d = _branches(study['panel_c'], x)
    h = _bias(np.column_stack((c, d)))
    target = np.column_stack((train['y'], train['traj']))
    trend = h @ _fit(h[:n], target)
    f = raw / raw[:n].std(axis=0)
    K = np.exp(-0.125*cdist(f, f[:n], 'sqeuclidean'))
    pred = trend[n:] + K[n:] @ np.linalg.solve(K[:n]+0.0001*np.eye(n), target-trend[:n])
    y, state = pred[:, 0], pred[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
