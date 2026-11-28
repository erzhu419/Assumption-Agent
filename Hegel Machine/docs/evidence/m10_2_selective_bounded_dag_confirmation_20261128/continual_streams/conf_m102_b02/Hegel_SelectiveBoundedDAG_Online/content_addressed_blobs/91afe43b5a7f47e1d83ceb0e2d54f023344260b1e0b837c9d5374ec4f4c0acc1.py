"""Recover latent coordinates and compose the active calibration panels."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _undo_sensor(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    x = np.empty_like(observed, dtype=float)
    for j in range(u.shape[1]):
        design = np.column_stack((np.ones(len(u)), u[:, j]))
        keep = np.ones(len(u), dtype=bool)
        for _ in range(8):
            coeff = _fit(design[keep], v[keep, j])
            residual = np.abs(design @ coeff - v[:, j])
            active = residual < max(0.0006, 4 * np.median(residual))
            if active.sum() < 3 or np.array_equal(active, keep):
                break
            keep = active
        offset, gain = _fit(design[keep], v[keep, j])
        x[:, j] = (observed[:, j] - offset) / gain
    return x


def _b_design(z, theta):
    z3 = z**3
    t = np.exp(theta[0]) * z + theta[1] * z3
    warped = t / np.hypot(1., t) + theta[2] * z + theta[3] * z3
    return np.column_stack((np.ones(len(z)), warped))


def _apply_b(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def residual(theta):
        d = _b_design(u, theta)
        return (d @ _fit(d, v) - v).ravel()

    fits = [least_squares(residual, start, bounds=([-3, -12, -12, -12], [2, 12, 12, 12]), max_nfev=150)
            for start in ([.4, 0., 0., 0.], [.5, 2., -2., -.5], [-.2, -.4, .3, .1])]
    theta = min(fits, key=lambda fit: fit.cost).x
    return _b_design(x, theta) @ _fit(_b_design(u, theta), v)


def _c_design(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), a*b, a*d, b*c, c*d))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    x = _undo_sensor(study['panel_a'], np.concatenate((train, test), axis=0))
    z = _apply_b(study['panel_b'], x)
    panel = study['panel_c']
    c = _c_design(z) @ _fit(_c_design(np.asarray(panel['u'], dtype=float)), np.asarray(panel['v1'], dtype=float))
    n = len(train)
    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])
