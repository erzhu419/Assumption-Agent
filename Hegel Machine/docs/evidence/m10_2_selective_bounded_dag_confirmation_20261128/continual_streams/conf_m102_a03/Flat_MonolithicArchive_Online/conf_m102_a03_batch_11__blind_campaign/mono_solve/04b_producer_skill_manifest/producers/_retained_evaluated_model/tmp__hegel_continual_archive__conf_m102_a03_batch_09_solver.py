"""Robust inverse calibration, an affine panel operator, and compact readouts."""
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


def _readout(w):
    return np.column_stack((_affine(w), w[:, 0] * w[:, 2] + w[:, 1] * w[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    cal = _calibration(study['panel_a'])
    z = _decode(train['x'], cal)
    zt = _decode(study['test']['x'], cal)
    panel = study['panel_b']
    operator = _fit(_affine(panel['u']), panel['v'])
    w = _affine(z) @ operator
    wt = _affine(zt) @ operator
    y_hat = _readout(wt) @ _fit(_readout(w), train['y'])

    features = _affine(w)
    slopes = _fit(features, train['traj'])
    # Retain the two dominant temporal modes of the fitted trajectories.
    _, _, modes = np.linalg.svd(features @ slopes, full_matrices=False)
    basis = modes[:2].T
    state_pred = (_affine(wt) @ slopes @ basis) @ basis.T
    last = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
