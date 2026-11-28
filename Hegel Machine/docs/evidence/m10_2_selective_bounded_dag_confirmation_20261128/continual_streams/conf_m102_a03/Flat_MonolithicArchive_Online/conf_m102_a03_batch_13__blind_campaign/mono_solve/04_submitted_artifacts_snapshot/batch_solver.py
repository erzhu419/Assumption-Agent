"""Robust cyclic calibration, cosine-panel reconstruction, and two output readouts."""
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



def _cosine_operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(frequency):
        return _affine(np.cos(frequency * u))
    def residual(p):
        a = design(p[0])
        return (a @ _fit(a, v) - v).ravel()
    grid = np.linspace(0.4, 6.0, 57)
    costs = [np.linalg.norm(residual([f])) for f in grid]
    f = least_squares(residual, [grid[np.argmin(costs)]], bounds=(0.1, 8.),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12).x[0]
    return f, _fit(design(f), v)


def _scalar_design(w):
    return np.column_stack((_affine(w), w[:, 0]*w[:, 2] + w[:, 1]*w[:, 3]))


def _scalar_readout(a, at, y):
    initial = _fit(a, y)
    def design(matrix, p):
        return np.column_stack((matrix, np.tanh(matrix @ p)))
    def residual(p):
        d = design(a, p)
        return d @ _fit(d, y) - y
    fit = min((least_squares(residual, initial * scale, max_nfev=160)
               for scale in (0.55, 1.15, 1.8)), key=lambda r:r.cost)
    return design(at, fit.x) @ _fit(design(a, fit.x), y)


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    frequency, mixing = _cosine_operator(study['panel_b'])
    def features(x):
        z = _decode(np.asarray(x, dtype=float), calibration)
        return _affine(np.cos(frequency * z)) @ mixing
    w, wt = (features(x) for x in (train['x'], study['test']['x']))
    y_hat = _scalar_readout(_scalar_design(w), _scalar_design(wt), np.asarray(train['y']))
    state_pred = _affine(wt) @ _fit(_affine(w), np.asarray(train['traj']))
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
