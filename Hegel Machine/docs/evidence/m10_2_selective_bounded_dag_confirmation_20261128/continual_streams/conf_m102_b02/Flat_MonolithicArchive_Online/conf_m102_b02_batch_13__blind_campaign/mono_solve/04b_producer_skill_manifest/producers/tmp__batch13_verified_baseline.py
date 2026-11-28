"""Recover a calibrated cosine–saturation graph and fit its bilinear readout."""
import numpy as np
from scipy.optimize import least_squares


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    slopes = []
    offsets = []
    for a, b in zip(u.T, v.T):
        i, j = np.triu_indices(len(a), 1)
        use = np.abs(a[i] - a[j]) > 0.05
        i, j = i[use], j[use]
        slope = np.median((b[i] - b[j]) / (a[i] - a[j]))
        offset = np.median(b - slope * a)
        error = b - offset - slope * a
        keep = np.abs(error) < max(0.001, 8 * np.median(np.abs(error)))
        if np.count_nonzero(keep) >= 3:
            offset, slope = _ls(_intercept(a[keep]), b[keep])
        slopes.append(slope)
        offsets.append(offset)
    return (x - offsets) / slopes


def _operator(panel, key, design, start, bounds, x):
    u, v = panel['u'], panel[key]

    def residual(p):
        a = design(u, p[0])
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [start], bounds=bounds, ftol=1e-12).x[0]
    return design(x, p) @ _ls(design(u, p), v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    w = _operator(study['panel_b'], 'v',
                  lambda a, q: _intercept(np.cos(q * a)), 2.6, (1.5, 3.5), z)
    h = _operator(study['panel_c'], 'v1',
                  lambda a, q: _intercept(np.column_stack((a, a / (1 + q * np.abs(a))))),
                  0.83, (0.2, 2.5), w)
    products = (z[:, :, None] * h[:, None, :]).reshape(len(z), 16)
    return _intercept(np.column_stack((z, h, products)))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    design = _features(study, x)
    n = len(train['x'])
    _, _, vh = np.linalg.svd(train['traj'], full_matrices=False)
    basis = vh[:2]
    targets = np.column_stack((train['y'], train['traj'] @ basis.T))
    pred = design[n:] @ _ls(design[:n], targets)
    y_hat, state_pred = pred[:, 0], pred[:, 1:] @ basis
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
