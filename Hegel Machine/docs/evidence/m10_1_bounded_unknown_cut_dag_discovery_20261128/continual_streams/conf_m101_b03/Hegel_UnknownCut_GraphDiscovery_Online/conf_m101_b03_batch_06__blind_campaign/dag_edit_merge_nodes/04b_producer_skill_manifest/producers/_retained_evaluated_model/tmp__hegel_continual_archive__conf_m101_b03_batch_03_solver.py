"""Predict batch responses from calibrated latent coordinates and operator interactions."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _latent(panel, x):
    u, v = panel['u'], panel['v']
    i = np.arange(u.shape[1])
    j = (i + 1) % len(i)
    design = np.stack((np.ones_like(u), u, u[:, j], u*u[:, j], u**3), axis=-1)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = np.stack([_fit(design[keep, k], v[keep, k]) for k in i], axis=1)
        residual = np.linalg.norm(np.sum(design*w.T, axis=-1)-v, axis=1)
        updated = residual <= max(4*np.median(residual), 1e-10)
        if np.array_equal(keep, updated):
            break
        keep = updated
    shift, gain, cross, product, cubic = w
    z = (x-shift)/gain
    for _ in range(24):
        next_z = z[:, j]
        residual = shift + gain*z + cross*next_z + product*z*next_z + cubic*z**3 - x
        if np.max(np.abs(residual), initial=0) < 1e-12:
            break
        jacobian = np.zeros((len(z), len(i), len(i)))
        jacobian[:, i, i] = gain + product*next_z + 3*cubic*z*z
        jacobian[:, i, j] = cross + product*z
        z -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    return z


def _operators(panel, z):
    u, v = panel['u'], panel['v1']

    def odd(x, p):
        rate, scale, weight = p
        return _affine(np.tanh(rate*x) + weight*x/(1+(scale*x)**2))

    def residual(p):
        d = odd(u, p)
        return (d@_fit(d, v)-v).ravel()

    fits = (least_squares(residual, p, bounds=([.2,.2,-30],[16,12,30]), max_nfev=180)
            for p in ((6,4,-5),(2,1,.5),(5,2,-4),(8,7,-4)))
    p = min(fits, key=lambda r:r.cost).x
    c = odd(z, p) @ _fit(odd(u, p), v)
    d = _affine(np.cos(2.4*z)) @ _fit(_affine(np.cos(2.4*u)), panel['v2'])
    return c, d


def _outer(a, b):
    return (a[:, :, None]*b[:, None, :]).reshape(len(a), a.shape[1]*b.shape[1])


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']))
    z = _latent(study['panel_a'], x)
    c, d = _operators(study['panel_c'], z)
    cyclic = np.sum(z * np.roll(c, 1, axis=1), axis=1)
    base = np.column_stack((np.ones(len(z)), c, d, cyclic))
    scalar = np.column_stack((base, np.sum(c*d, axis=1)))
    state = np.column_stack((base, _outer(c, d)))
    y_hat = scalar[n:] @ _fit(scalar[:n], tr['y'])
    state_pred = state[n:] @ _fit(state[:n], tr['traj'])
    endpoint = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
