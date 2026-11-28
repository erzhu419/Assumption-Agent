"""Predict through the inferred calibration, cosine and mixed activation path."""
import numpy as np
from scipy.optimize import least_squares, root


def _design(*parts):
    return np.column_stack((np.ones(len(parts[0])), *parts))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _latent(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def correction(w):
        return .18 * np.tanh(1.8 * w) - .32 * np.sin(2.4 * np.roll(w, 2, axis=-1))

    target = v - correction(u)
    keep = np.max(np.abs(v-u), axis=1) < 3
    for _ in range(8):
        offset, scale = np.array([_ls(_design(u[keep, j]), target[keep, j])
                                  for j in range(u.shape[1])]).T
        err = np.linalg.norm(u * scale + offset - target, axis=1)
        next_keep = err < max(.001, 4 * np.median(err))
        if np.array_equal(next_keep, keep) or next_keep.sum() < len(u)//2:
            break
        keep = next_keep
    out = []
    for row in x:
        initial = (row-offset)/scale
        def residual(z):
            return z*scale+offset+correction(z)-row
        ans = root(residual, initial, tol=1e-10)
        if np.linalg.norm(residual(ans.x), ord=np.inf) > 1e-8:
            ans = least_squares(residual, initial, gtol=1e-12)
        out.append(ans.x)
    return np.asarray(out)


def _operator(panel, key, basis, start, bounds, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', key))

    def residual(q):
        a = basis(u, q)
        return (a @ _ls(a, v)-v).ravel()

    q = least_squares(residual, start, bounds=bounds, gtol=1e-11).x
    return basis(x, q) @ _ls(basis(u, q), v)


def _features(study, x):
    z = _latent(study['panel_a'], x)
    w = _operator(study['panel_b'], 'v', lambda a,q: _design(np.cos(q[0]*a)),
                  [2.6], ([1.5], [3.5]), z)
    h = _operator(study['panel_c'], 'v1',
                  lambda a,q: _design(np.tanh(q[0]*a), np.sin(q[1]*a)),
                  [1.75, 2.1], ([1, 1.7], [2.5, 2.6]), w)
    readout = _design(z, h, np.sum(z*h, axis=1))
    return readout, np.column_stack((readout, z[:, 1]*h[:, 0]))


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']), axis=0)
    scalar, temporal = _features(study, x)
    y = scalar[n:] @ _ls(scalar[:n], tr['y'])
    obs = np.asarray(tr['traj'])
    modes = np.linalg.svd(obs, full_matrices=False)[2][:2]
    state = (temporal[n:] @ _ls(temporal[:n], obs @ modes.T)) @ modes
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
