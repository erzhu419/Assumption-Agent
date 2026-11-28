"""Fit the calibrated two-branch response and two-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _align(panel):
    u, v = panel['u'], panel['v']
    # Median of pairwise slopes, followed by inlier least squares.
    i, j = np.triu_indices(len(u), 1)
    dx, dy = u[i] - u[j], v[i] - v[j]
    result = []
    for k in range(u.shape[1]):
        valid = abs(dx[:, k]) > 1e-8
        slope = np.median(dy[valid, k] / dx[valid, k])
        r = v[:, k] - u[:, k] * slope
        r -= np.median(r)
        good = abs(r) <= max(.001, 5 * np.median(abs(r)))
        result.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    return np.asarray(result).T


def _branch(panel, x, k):
    u, v = panel['u'], panel['v'+str(k)]
    if k == 1:
        def features(z, p):
            w = p[0]*z + p[1:]
            return _ones(np.c_[z, w/(1+abs(w))])
        start = [.84, 0., 0., 0., 0.]
    else:
        def features(z, p):
            return _ones(np.c_[z, np.cos(p[0]*z+p[1:])])
        start = [1.27, 0., 0., 0., 0.]
    def residual(p):
        d = features(u, p)
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, start, max_nfev=150).x
    return features(x, p) @ _ls(features(u, p), v)


def _predict(study, rows, xnew):
    train = study['train']
    xold = train['x'][rows]
    n = len(xold)
    a, b = _align(study['panel_a'])
    z = (np.vstack((xold, xnew)) - a) / b
    c, d = (_branch(study['panel_c'], z, k) for k in (1,2))
    dot = np.sum(c*d, axis=1)
    sy = _ones(np.c_[d, c, dot])
    syhat = sy[n:] @ _ls(sy[:n], train['y'][rows])
    ts = np.asarray(study['eval_times'])
    trajectories = train['traj'][rows]
    def basis(p):
        return -np.expm1(-ts[:, None]*p)
    def time_residual(p):
        h = basis(p)
        return (h @ _ls(h, trajectories.T) - trajectories.T).ravel()
    h = basis(least_squares(time_residual, [1.8,.4], bounds=([1.,.1],[3.,.95])).x)
    cross = (d[:,:,None]*c[:,None,:]).reshape(len(z), -1)
    st = _ones(np.c_[d,c,cross])
    return syhat, st[n:] @ _ls(st[:n], trajectories) @ np.linalg.pinv(h.T) @ h.T


def solve_batch(study: dict) -> dict:
    y, state = _predict(study, slice(None), study['test']['x'])
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
