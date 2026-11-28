"""Recover latent coordinates and calibrate two response operators."""
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
    u = np.asarray(panel['u'], dtype=float)

    def soft(a, q):
        return _one(np.c_[a, a*np.abs(a), a/(1+q*np.abs(a))])

    def rational(a, q):
        return _one(np.c_[a, a*np.roll(a, -1, axis=1), a/(1+q*a*a)])

    def project(basis, target):
        def loss(q):
            f = basis(u, q)
            residual = f @ _ls(f, target) - target
            return np.sum(residual*residual)
        q = minimize_scalar(loss, bounds=(.05, 2.5), method='bounded',
                            options={'xatol': 1e-9}).x
        return basis(z, q) @ _ls(basis(u, q), target)

    return project(soft, panel['v1']), project(rational, panel['v2'])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.vstack((study['train']['x'], study['test']['x']))
    z = _coords(study['panel_a'], x)
    c, d = _operators(study['panel_c'], z)
    return {'features': np.column_stack((z, c, d)),
            'calibrated': np.column_stack((c, d))}
