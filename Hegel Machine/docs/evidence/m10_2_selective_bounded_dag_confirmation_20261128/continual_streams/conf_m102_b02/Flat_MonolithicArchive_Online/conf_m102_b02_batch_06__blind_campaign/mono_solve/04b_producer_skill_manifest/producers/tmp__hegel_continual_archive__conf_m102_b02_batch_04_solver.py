"""Fit the calibrated panel composition and its response and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _intercept(a):
    return np.column_stack((np.ones(len(a)), a))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        center = u[keep].mean(axis=0)
        du = u[keep] - center
        scale = (du * (v[keep] - v[keep].mean(axis=0))).sum(axis=0) / (du**2).sum(axis=0)
        shift = v[keep].mean(axis=0) - center * scale
        residual = np.linalg.norm(v - shift - u * scale, axis=1)
        next_keep = residual <= max(.001, 3 * np.median(residual))
        if np.array_equal(keep, next_keep) or next_keep.sum() < 8:
            break
        keep = next_keep
    return (x - shift) / scale


def _panel_b(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(a, q):
        return _intercept(np.column_stack((a, np.sin(a * np.exp(q[0])))))
    def residual(q):
        d = design(u, q)
        return (d @ _ls(d, v) - v).ravel()
    q = min((least_squares(residual, [a], bounds=(-2.5, 1.6))
             for a in (-.3, .5, 1.15)), key=lambda fit: fit.cost).x
    return design(z, q) @ _ls(design(u, q), v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    w = _panel_b(study['panel_b'], z)
    panel = study['panel_c']
    def adjacent(a):
        return _intercept(a * np.roll(a, -1, axis=1))
    b = adjacent(w) @ _ls(adjacent(panel['u']), panel['v1'])
    base = _intercept(np.column_stack((z, b)))
    cross = z[:, :, None] * b[:, None, :]
    diagonal = np.diagonal(cross, axis1=1, axis2=2)
    response = np.column_stack((base, diagonal.sum(axis=1)))
    # Retain the diagonal couplings and the supported z[1]–b[0] interaction.
    state = np.column_stack((base, diagonal, cross[:, 1, 0]))
    return response, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    fy, fs = _features(study, np.concatenate((train['x'], study['test']['x'])))
    y_hat = fy[n:] @ _ls(fy[:n], train['y'])
    trajectory = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(trajectory, full_matrices=False)[2][:2]
    state_pred = (fs[n:] @ _ls(fs[:n], trajectory @ modes.T)) @ modes
    last = state_pred[:, -1]
    return dict(y_hat=y_hat, state_pred=state_pred,
                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))
