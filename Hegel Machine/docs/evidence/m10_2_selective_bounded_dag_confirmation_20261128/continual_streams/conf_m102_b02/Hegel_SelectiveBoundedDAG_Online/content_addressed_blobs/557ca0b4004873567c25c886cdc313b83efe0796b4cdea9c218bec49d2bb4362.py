"""Fit the spatial calibration chain and construct the bilinear readout design."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibrate(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    result = np.empty_like(measured, dtype=float)
    for j in range(u.shape[1]):
        d = _affine(u[:, j])
        y = v[:, j]
        keep = np.ones(len(u), dtype=bool)
        for _ in range(8):
            coef = np.linalg.lstsq(d[keep], y[keep], rcond=None)[0]
            residual = np.abs(y - d @ coef)
            updated = residual <= max(0.0006, 4 * np.median(residual))
            if updated.sum() < 3 or np.array_equal(updated, keep):
                break
            keep = updated
        coef = np.linalg.lstsq(d[keep], y[keep], rcond=None)[0]
        result[:, j] = (measured[:, j] - coef[0]) / coef[1]
    return result


def _profile(d, target):
    coef = np.linalg.lstsq(d, target, rcond=None)[0]
    return (d @ coef - target).ravel(), coef


def _panel_b(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(q, freq):
        return _affine(1 - np.cos(freq * q))

    def objective(freq):
        r, _ = _profile(design(u, freq), v)
        return r @ r

    freq = minimize_scalar(objective, bounds=(0.5, 4.0), method='bounded',
                           options={'xatol': 1e-10}).x
    return design(x, freq) @ _profile(design(u, freq), v)[1]


def _panel_c(panel, b):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1'))

    def design(q, p):
        a = np.abs(q)
        return _affine(q * a / (1 + np.exp(p[0]) * a) + p[1] * q**3 + p[2] * q)

    params = least_squares(lambda p: _profile(design(u, p), v)[0],
                           [-0.42, -0.02, -1.08], max_nfev=200).x
    return design(b, params) @ _profile(design(u, params), v)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    n = len(train)
    x = _calibrate(study['panel_a'], np.concatenate((train, test)))
    b = _panel_b(study['panel_b'], x)
    c = _panel_c(study['panel_c'], b)
    sx, sc = x[:n].std(axis=0), c[:n].std(axis=0)
    x /= np.where(sx > 1e-12, sx, 1.0)
    c /= np.where(sc > 1e-12, sc, 1.0)
    interaction = (x[:, :, None] * c[:, None, :]).reshape(len(x), -1)
    d = _affine(np.column_stack((x, c, interaction)))
    return {'train_design': d[:n], 'test_design': d[n:]}
