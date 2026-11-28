"""Infer the calibrated panel composition and fit separate outcome readouts."""
import numpy as np
from scipy.optimize import least_squares


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _coordinates(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    a = _design(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        c = _ls(a[keep], v[keep])
        err = np.linalg.norm(a @ c - v, axis=1)
        new = err < max(0.001, 4 * np.median(err))
        if new.sum() < len(c) + 2 or np.array_equal(new, keep):
            break
        keep = new
    c = _ls(a[keep], v[keep])
    return np.linalg.solve(c[1:].T, (x - c[0]).T).T


def _features(study, x):
    z = _coordinates(study['panel_a'], x)
    b = study['panel_b']
    w = _design(z) @ _ls(_design(b['u']), b['v'])
    c = study['panel_c']
    u, v = c['u'], c['v1']

    def operator_basis(a, q):
        return _design(np.column_stack((a, np.sin(q[0] * a), np.tanh(q[1] * a))))

    def residual(q):
        a = operator_basis(u, q)
        return (a @ _ls(a, v) - v).ravel()

    q = least_squares(residual, [2.1, 1.75], bounds=([0.5, 0.5], [4.5, 5.0]), gtol=1e-11).x
    h = operator_basis(w, q) @ _ls(operator_basis(u, q), v)
    y_features = _design(np.column_stack((z, h, np.sum(z * h, axis=1))))
    state_features = np.column_stack((y_features, z[:, 1] * h[:, 0]))
    return y_features, state_features


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    fy, ft = _features(study, np.concatenate((train['x'], study['test']['x'])))
    y = fy[n:] @ _ls(fy[:n], train['y'])
    traj = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state = (ft[n:] @ _ls(ft[:n], traj @ modes.T)) @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
