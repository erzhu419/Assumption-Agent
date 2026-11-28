"""Batch prediction using calibrated polynomial and cosine operator branches."""
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(u - v, axis=1) < 1.0
    if keep.sum() < 8:
        keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        offset, scale = np.array([
            _fit(_affine(u[keep, j:j+1]), v[keep, j]) for j in range(u.shape[1])
        ]).T
        error = np.linalg.norm(u * scale + offset - v, axis=1)
        new = error <= max(1e-7, 5.0 * np.median(error))
        if new.sum() < 8 or np.array_equal(new, keep):
            break
        keep = new
    return (x - offset) / scale


def _products(q):
    return np.column_stack([q[:, i] * q[:, j] for i, j in ((0, 1), (0, 3), (1, 2), (2, 3))])


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), a.shape[1] * b.shape[1])


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    q = _calibrate(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    panel = study['panel_c']
    c = _affine(_products(q)) @ _fit(_affine(_products(panel['u'])), panel['v1'])
    # The cosine coordinates span panel C's second branch after its affine mixing.
    d = np.cos(2.4 * q)
    features = np.column_stack((q, c, d, _cross(q, c), _cross(c, d)))
    center = features[:n].mean(axis=0)
    scale = np.maximum(features[:n].std(axis=0), 1e-8)
    features = (features - center) / scale
    traj = np.asarray(train['traj'])
    basis = np.linalg.svd(traj, full_matrices=False)[2][:3]
    target = np.column_stack((train['y'], traj @ basis.T))
    mean = target.mean(axis=0)
    a = features[:n]
    weights = np.linalg.solve(a.T @ a + 0.002 * np.eye(a.shape[1]), a.T @ (target - mean))
    pred = features[n:] @ weights + mean
    y = pred[:, 0]
    state = pred[:, 1:] @ basis
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
