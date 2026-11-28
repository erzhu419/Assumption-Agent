"""Calibrated two-operator readout with a two-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _one(a):
    return np.column_stack((np.ones(len(a)), a))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _coords(panel, x):
    u, v = panel['u'], panel['v']
    j = np.arange(u.shape[1]); nxt = (j+1) % len(j)

    def calibration_term(z):
        return .32*z[:, nxt]*(z-1) + .16*z**3

    adj = v-calibration_term(u)
    slope = np.empty(len(j))
    for k in j:
        a, b = u[:, k], adj[:, k]
        ii, jj = np.triu_indices(len(a), 1)
        diff = a[ii]-a[jj]
        good = np.abs(diff)>1e-8
        slope[k] = np.median((b[ii]-b[jj])[good]/diff[good])
    offset = np.median(adj-u*slope, axis=0)
    for _ in range(4):
        residual = np.abs(adj-offset-u*slope)
        for k in j:
            keep = residual[:, k] <= max(.0005, 6*np.median(residual[:, k]))
            offset[k], slope[k] = _ls(_one(u[keep, k]), adj[keep, k])

    def forward(z):
        return calibration_term(z)+offset+slope*z

    z0 = (x-offset)/slope
    z = z0.copy()
    for _ in range(32):
        h = forward(z)-x
        if np.max(np.abs(h), initial=0) < 1e-10:
            return z
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = slope+.32*z[:, nxt]+.48*z*z
        jac[:, j, nxt] = .32*(z-1)
        step = np.linalg.solve(jac, h[..., None])[..., 0]
        z -= step/np.maximum(1, np.linalg.norm(step, axis=1, keepdims=True))
    for i in np.flatnonzero(np.max(np.abs(forward(z)-x), axis=1)>1e-8):
        z[i] = least_squares(lambda q: forward(q[None])[0]-x[i], z0[i]).x
    return z


def _operators(panel, z):
    u = panel['u']
    def c_basis(a, q):
        return _one(np.c_[a, a**3, np.arctan(q*a)])
    def loss(q):
        f = c_basis(u, q)
        return np.linalg.norm(f@_ls(f, panel['v1'])-panel['v1'])
    q = minimize_scalar(loss, bounds=(.45, 3.1), method='bounded').x
    c = c_basis(z, q)@_ls(c_basis(u, q), panel['v1'])
    d = _one(np.cos(2.4*z))@_ls(_one(np.cos(2.4*u)), panel['v2'])
    return c, d


def _design(c, d):
    return _one(np.c_[c, d, np.sum(c*d, axis=1), c[:, 0]*d[:, 1]])


def _time(t, traj):
    def basis(rate):
        return -np.expm1(-t[:, None]*rate)
    def residual(rate):
        f = basis(rate)
        return (f@_ls(f, traj.T)-traj.T).ravel()
    q = least_squares(residual, [.4, 1.8], bounds=([.04,.8],[.8,4.])).x
    return basis(q)


def solve_batch(study: dict) -> dict:
    train = study['train']; n = len(train['y'])
    z = _coords(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    design = _design(*_operators(study['panel_c'], z))
    q = _time(study['eval_times'], train['traj'])
    load = _ls(q, train['traj'].T).T
    scalar_design = design[:, :-1]
    y = scalar_design[n:]@_ls(scalar_design[:n], train['y'])
    state = design[n:]@_ls(design[:n], load)@q.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
