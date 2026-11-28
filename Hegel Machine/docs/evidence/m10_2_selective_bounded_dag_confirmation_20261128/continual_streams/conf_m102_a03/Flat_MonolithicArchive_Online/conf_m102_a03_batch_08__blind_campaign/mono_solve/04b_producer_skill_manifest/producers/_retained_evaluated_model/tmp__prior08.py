"""Calibrated dead-zone features for scalar and trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n = len(u)
    a = u.ravel()
    b = np.roll(u, -1, axis=1).ravel()
    channel = np.tile(np.eye(4), (n, 1))
    design = np.column_stack((channel, channel * a[:, None], b, a*b, a**3))
    discrepancy = np.linalg.norm(v-u, axis=1)
    good = discrepancy < 4 * max(np.median(discrepancy), 1e-8)
    if good.sum() < 8:
        good[:] = True
    for _ in range(5):
        keep = np.repeat(good, 4)
        coef = _fit(design[keep], v.ravel()[keep])
        residual = np.linalg.norm((design @ coef).reshape(n, 4) - v, axis=1)
        refined = residual <= max(5 * np.median(residual), 1e-7)
        if refined.sum() < 8 or np.array_equal(good, refined):
            break
        good = refined
    return coef


def _decode(x, coef):
    offset, gain = coef[:4], coef[4:8]
    coupling, interaction, cubic = coef[8:]
    i = np.arange(4)
    j = (i + 1) % 4

    def residual(z, row):
        nxt = z[j]
        return offset + gain*z + coupling*nxt + interaction*z*nxt + cubic*z**3 - row

    def jacobian(z, row):
        g = np.diag(gain + interaction*z[j] + 3*cubic*z*z)
        g[i, j] = coupling + interaction*z
        return g

    z = []
    for row in x:
        result = least_squares(residual, (row-offset)/gain, jac=jacobian,
                               args=(row,), gtol=1e-12, max_nfev=80)
        if np.linalg.norm(result.fun, np.inf) > 1e-8:
            other = least_squares(residual, np.zeros(4), jac=jacobian,
                                  args=(row,), gtol=1e-12, max_nfev=100)
            if np.linalg.norm(other.fun) < np.linalg.norm(result.fun):
                result = other
        z.append(result.x)
    return np.asarray(z).reshape(-1, 4)


def _warp(x, p):
    return np.sign(x) * np.maximum(np.abs(x) - p[0], 0.0) ** p[1]


def _operator(panel):
    u, v = [np.asarray(panel[k], dtype=float) for k in ('u', 'v')]

    def residual(p):
        a = _affine(_warp(u, p))
        return (a @ _fit(a, v) - v).ravel()

    scale = np.max(np.abs(u))
    p = least_squares(residual, [0.25 * scale, 1.5],
                      bounds=([0.0, 0.5], [0.8 * scale, 3.5]),
                      ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return p, _fit(_affine(_warp(u, p)), v)


from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar


def _features(z, w):
    base = np.column_stack((_affine(w), z * np.roll(w, 1, axis=1)))
    quad = np.column_stack([w[:, i]*w[:, j] for i, j in combinations_with_replacement(range(4), 2)])
    return base, quad


def _scalar(z, w, nt, y):
    base, quad = _features(z, w)
    d = z[:, 3]
    # Saturated direct branch and unsaturated cyclic and operator-pair terms.
    def fit(q, predict=False):
        den = 1 + q*d*d
        design = np.c_[base[:, :5]/den[:, None], base[:, 5:], quad, d, d/den]
        train = design[:nt]
        coef = _fit(train, y)
        return design[nt:] @ coef if predict else np.sum((train @ coef - y)**2)
    q = minimize_scalar(fit, bounds=(.025, .8), method='bounded').x
    return fit(q, True)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    cal = _calibration(study['panel_a'])
    z = _decode(np.concatenate((train['x'], study['test']['x'])), cal)
    p, b = _operator(study['panel_b'])
    w = _affine(_warp(z, p)) @ b
    y_hat = _scalar(z, w, n, np.asarray(train['y'], dtype=float))
    basis, _ = _features(z, w)
    scale = np.maximum(np.sqrt(np.mean(basis[:n]**2, axis=0)), 1e-7)
    a = basis[:n]/scale
    state = (basis[n:]/scale) @ np.linalg.solve(a.T@a + .001*np.eye(a.shape[1]), a.T@train['traj'])
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
