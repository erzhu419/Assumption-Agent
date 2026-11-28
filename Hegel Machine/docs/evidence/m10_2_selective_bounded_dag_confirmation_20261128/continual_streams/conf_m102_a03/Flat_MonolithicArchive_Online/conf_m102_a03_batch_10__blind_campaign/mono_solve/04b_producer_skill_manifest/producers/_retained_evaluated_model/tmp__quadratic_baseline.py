"""Predict the scalar and trajectory from a calibrated dead-zone operator."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = u.ravel()
    b = np.roll(u, 2, axis=1).ravel()
    channel = np.tile(np.eye(4), (len(u), 1))

    def design(p):
        return np.column_stack((channel, channel * a[:, None],
                                np.tanh(p[0] * a), np.sin(p[1] * b)))

    discrepancy = np.linalg.norm(v - u, axis=1)
    mask = discrepancy <= 4 * max(np.median(discrepancy), 1e-8)
    if mask.sum() < 8:
        mask[:] = True
    p = np.array([1.8, 2.4])
    for _ in range(5):
        keep = np.repeat(mask, 4)
        target = v.ravel()[keep]

        def residual(q):
            f = design(q)[keep]
            return f @ _fit(f, target) - target

        p = least_squares(residual, p, bounds=([0.5, 1.5], [3.5, 3.5]),
                          ftol=1e-12, gtol=1e-12, xtol=1e-12).x
        f = design(p)
        c = _fit(f[keep], target)
        err = np.linalg.norm((f @ c).reshape(v.shape) - v, axis=1)
        updated = err <= max(6 * np.median(err), 1e-7)
        if updated.sum() < 8 or np.array_equal(updated, mask):
            break
        mask = updated
    return p, c


def _decode(x, cal):
    p, c = cal
    bias, gain = c[:4], c[4:8]
    h, k = c[8:]
    i = np.arange(4)
    j = (i + 2) % 4

    def residual(z, x):
        return bias + gain * z + h * np.tanh(p[0] * z) + k * np.sin(p[1] * z[j]) - x

    def jac(z, x):
        q = np.tanh(p[0] * z)
        g = np.diag(gain + h * p[0] * (1 - q*q))
        g[i, j] = k * p[1] * np.cos(p[1] * z[j])
        return g

    z = [least_squares(residual, (row - bias) / gain, jac=jac, args=(row,),
                       gtol=1e-12, ftol=1e-12, xtol=1e-12).x for row in x]
    return np.asarray(z).reshape(-1, 4)


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))



def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(x, p):
        return _affine(np.sign(x) * np.maximum(np.abs(x) - p[0], 0.) ** p[1])
    def residual(p):
        a = design(u, p)
        return (a @ _fit(a, v) - v).ravel()
    p = least_squares(residual, [.35, 1.5], bounds=([0., .2], [1., 5.])).x
    c = _fit(design(u, p), v)
    return lambda x: design(x, p) @ c


def _quadratic(w):
    i, j = np.triu_indices(w.shape[1])
    return np.column_stack((w, w[:, i] * w[:, j]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    cal = _calibration(study['panel_a'])
    encode = _operator(study['panel_b'])
    w = encode(_decode(train['x'], cal))
    wt = encode(_decode(study['test']['x'], cal))
    a, b = _quadratic(w), _quadratic(wt)
    scale = np.maximum(a.std(axis=0), .01)
    a, b = _affine(a / scale), _affine(b / scale)
    penalty = np.eye(a.shape[1]) * .02
    penalty[0, 0] = 0.
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ np.column_stack((train['y'], train['traj'])))
    pred = b @ coef
    y, state = pred[:, 0], pred[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
