"""Cyclic calibration, threshold encoding, bilinear readout, and two-rate relaxation."""
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


def _readout(w):
    return np.column_stack((w, w[:, 0] * w[:, 2] + w[:, 1] * w[:, 3]))


def _relaxation(w, wt, observed, times):
    # Profile out the spatial coefficients before fitting the two decay rates.
    q = np.linalg.qr(w, mode='reduced')[0]
    target = (q.T @ observed).T

    def basis(p):
        return -np.expm1(-np.outer(times, p))

    def residual(p):
        b = basis(p)
        return (b @ _fit(b, target) - target).ravel()

    rate = least_squares(residual, [0.4, 1.8],
                         bounds=([0.005, 0.6], [0.8, 25.0]),
                         ftol=1e-10, xtol=1e-10, gtol=1e-10).x
    b = basis(rate)
    amplitudes = _fit(b, observed.T).T
    return wt @ _fit(w, amplitudes) @ b.T


def solve_batch(study: dict) -> dict:
    """Fit the two readouts using reference panels and labeled training rows."""
    train = study['train']
    calibration = _calibration(study['panel_a'])
    params, mix = _operator(study['panel_b'])

    def encode(x):
        z = _decode(np.asarray(x, dtype=float), calibration)
        return _affine(_warp(z, params)) @ mix

    w = encode(train['x'])
    wt = encode(study['test']['x'])
    y_hat = _readout(wt) @ _fit(_readout(w), np.asarray(train['y'], dtype=float))
    state = _relaxation(w, wt, np.asarray(train['traj'], dtype=float),
                        np.asarray(study['eval_times'], dtype=float))
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
