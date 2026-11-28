"""Batch prediction with a calibrated spatial kernel and shared temporal modes."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def _calibration(panel):
    u, v = map(np.asarray, (panel['u'], panel['v']))
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slopes = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)
    offsets = np.median(v - u * slopes, axis=0)
    for _ in range(3):
        r = np.linalg.norm(v - u * slopes - offsets, axis=1)
        med = np.median(r)
        keep = r <= max(6 * med, 0.0005)
        if keep.sum() < 5:
            break
        for k in range(u.shape[1]):
            slopes[k], offsets[k] = np.linalg.lstsq(np.c_[u[keep, k], np.ones(keep.sum())], v[keep, k], rcond=None)[0]
    return np.where(abs(slopes) > 1e-6, slopes, 1.), offsets


def _kernel(x, q, slope):
    i, j = np.triu_indices(x.shape[1])
    return (1. + x @ q.T + .30 * ((x[:, i] * x[:, j]) @ (q[:, i] * q[:, j]).T)
            + 1.5 * np.exp(-cdist(x, q, 'sqeuclidean') / 6.)
            + 1.5 * np.exp(-cdist(x / slope, q / slope, 'sqeuclidean') / 6.))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    q = np.asarray(study['test']['x'], dtype=float)
    y, t = map(np.asarray, (train['y'], train['traj']))
    slope, _ = _calibration(study['panel_a'])
    _, _, modes = np.linalg.svd(t, full_matrices=False)
    modes = modes[:2]
    target = np.c_[y, t @ modes.T]
    mu = target.mean(axis=0)
    K = _kernel(x, x, slope)
    K.flat[::len(K)+1] += .0008
    p = _kernel(q, x, slope) @ solve(K, target - mu, assume_a='pos') + mu
    yh, st = p[:, 0].copy(), p[:, 1:] @ modes
    return {'y_hat': yh, 'state_pred': st,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), st[:, -1].mean(), st[:, -1].std()])}
