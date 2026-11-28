"""Calibrated multi-output regression with a shared low-rank state trajectory."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _coordinates(study, x):
    a = study['panel_a']
    A, v = _design(a['u']), a['v']
    keep = np.ones(len(A), dtype=bool)
    for _ in range(6):
        w = _fit(A[keep], v[keep])
        e = np.linalg.norm(A @ w - v, axis=1)
        good = e < max(.001, 3 * np.median(e))
        if np.array_equal(keep, good) or good.sum() < A.shape[1]:
            break
        keep = good
    z = np.linalg.solve(w[1:].T, (x - w[0]).T).T
    p = study['panel_b']
    u, v = p['u'], p['v']
    def phi(q, u):
        return _design(erf(q[0]*u) + q[1]*u)
    def residual(q):
        F = phi(q, u)
        return (F @ _fit(F, v) - v).ravel()
    sols = [least_squares(residual, [a, -.5*a], bounds=([.1,-8],[4,8]), max_nfev=80) for a in [.4,.8,1.4,2.2]]
    q = min(sols, key=lambda o:o.cost).x
    b = phi(q, z) @ _fit(phi(q, u), v)
    return z, b


def _kernel(x, w):
    linear = x @ w.T / x.shape[1]
    r = .5 * np.abs(x[:, None, :] - w[None, :, :])
    g = (1 + r + r*r/3)*np.exp(-r)
    avg = g.mean(2)
    pair = avg*avg - (g*g).sum(2)/x.shape[1]**2
    return (1+linear)**2 + 2*avg + 2*pair


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    z, b = _coordinates(study, np.vstack((tr['x'], study['test']['x'])))
    x = np.c_[z, b, np.tanh(2*z), np.tanh(2*b)]
    x = x / np.maximum(x[:n].std(0), 1e-10)
    i, j = x[:n], x[n:]
    traj = np.asarray(tr['traj'])
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    k, ktest = _kernel(i, i), _kernel(j, i)
    y = ktest @ np.linalg.solve(k + .001*np.eye(n), tr['y'])
    state = (ktest @ np.linalg.solve(k + .001*np.eye(n), traj @ modes.T)) @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
