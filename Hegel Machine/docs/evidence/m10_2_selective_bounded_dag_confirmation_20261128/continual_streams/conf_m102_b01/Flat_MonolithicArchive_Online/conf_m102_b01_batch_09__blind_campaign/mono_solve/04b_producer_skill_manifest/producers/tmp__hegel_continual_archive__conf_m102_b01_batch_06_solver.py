"""Panel-calibrated cyclic readouts for the scalar and the trajectory."""
import numpy as np
from scipy.optimize import least_squares, root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _cal_basis(z):
    nxt = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, nxt, z * nxt, z ** 3), axis=-1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _cal_basis(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        coef = np.array([_ls(d[keep, j], v[keep, j]) for j in range(4)])
        residual = np.linalg.norm((d * coef).sum(axis=-1) - v, axis=1)
        next_keep = residual <= max(0.001, 8 * np.median(residual))
        if np.array_equal(keep, next_keep):
            break
        keep = next_keep
    return coef


def _inverse(coef, x):
    def solve(row):
        def residual(z):
            return (_cal_basis(z) * coef).sum(axis=-1) - row
        fit = root(residual, row, tol=1e-10)
        if np.linalg.norm(fit.fun, ord=np.inf) > 1e-8:
            fit = least_squares(residual, fit.x, gtol=1e-12)
        return fit.x
    return np.array([solve(row) for row in x]).reshape(-1, 4)


def _dead(z):
    return np.sign(z) * np.maximum(np.abs(z) - 0.35, 0) ** 1.5


def _cross(z):
    return _aff(np.column_stack((z[:, 0] * z[:, 1], z[:, 0] * z[:, 3],
                                 z[:, 1] * z[:, 2], z[:, 2] * z[:, 3])))


def _coordinates(study, x):
    z = _inverse(_calibration(study['panel_a']), x)
    b, c = study['panel_b'], study['panel_c']
    w = _aff(_dead(z)) @ _ls(_aff(_dead(b['u'])), b['v'])
    q = _cross(w) @ _ls(_cross(c['u']), c['v1'])
    return z, w, q


def _designs(study, x):
    z, _, q = _coordinates(study, x)
    cyclic = q * np.roll(z, -1, axis=1)
    state = _aff(np.column_stack((q, cyclic.sum(axis=1))))
    # Bound the scalar features jointly as the leading cyclic product grows.
    attenuation = 1.0 / (1.0 + 0.5 * cyclic[:, :1] ** 2)
    scalar = _aff(np.column_stack((q, cyclic)) * attenuation)
    return state, scalar


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    state, scalar = _designs(study, np.concatenate((train['x'], study['test']['x'])))
    y = scalar[n:] @ _ls(scalar[:n], train['y'])
    p = state[n:] @ _ls(state[:n], train['traj'])
    tail = p[:, -1]
    return {'y_hat': y, 'state_pred': p,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
