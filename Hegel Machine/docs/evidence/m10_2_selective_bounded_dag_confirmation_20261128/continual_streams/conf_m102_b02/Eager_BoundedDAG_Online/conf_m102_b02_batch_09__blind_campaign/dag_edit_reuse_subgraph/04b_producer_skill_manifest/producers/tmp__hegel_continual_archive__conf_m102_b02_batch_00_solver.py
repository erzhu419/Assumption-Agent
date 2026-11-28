"""Predict the campaign outputs from calibrated, mixed input coordinates."""
import numpy as np


def _affine(panel, robust=False):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    design = np.column_stack((np.ones(len(u)), u))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6 if robust else 1):
        coef = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(design @ coef - v, axis=1)
        proposed = residual < max(0.001, 3 * np.median(residual))
        if not robust or np.array_equal(proposed, keep) or proposed.sum() < design.shape[1]:
            break
        keep = proposed
    return coef


def _kernel(z, w):
    h, g = np.arcsinh(3.5 * z), np.arcsinh(3.5 * w)
    linear = z @ w.T
    compressed = h @ g.T
    return (1 + linear + compressed + (h*h) @ (g*g).T
            + (z*h) @ (w*g).T + 0.2 * (compressed + 0.25*linear)**2)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    xt = np.asarray(study['test']['x'], dtype=float)
    a = _affine(study['panel_a'], robust=True)
    b = _affine(study['panel_b'])
    z = np.linalg.solve(a[1:].T, (np.vstack((x, xt)) - a[0]).T).T @ b[1:] + b[0]
    z, zt = z[:len(x)], z[len(x):]
    traj = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    targets = np.column_stack((train['y'], traj @ modes.T))
    gram = _kernel(z, z)
    gram.flat[::len(z)+1] += 1e-6 * np.mean(np.diag(gram))
    pred = _kernel(zt, z) @ np.linalg.solve(gram, targets)
    y = pred[:, 0].copy()
    state = pred[:, 1:] @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
