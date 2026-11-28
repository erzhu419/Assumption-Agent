"""Panel-informed scalar prediction and low-rank trajectory regression."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibration(study):
    a, b, c = (study['panel_' + k] for k in 'abc')
    u, v = a['u'], a['v']
    # The calibration is near identity except for grossly contaminated rows.
    coeff = []
    for j in range(u.shape[1]):
        keep = np.abs(v[:, j] - u[:, j]) < 1.5
        coeff.append(_ls(_affine(u[keep, j:j + 1]), v[keep, j]))
    bias, gain = np.array(coeff).T
    rotation = _ls(_affine(b['u']), b['v'])[1:]

    def basis(x, p):
        q = x * x
        return _affine(np.c_[x / (1 + np.exp(p[0]) * q), q / (1 + np.exp(p[1]) * q)])

    def residual(p):
        f = basis(c['u'], p)
        return (f @ _ls(f, c['v2']) - c['v2']).ravel()

    p = least_squares(residual, [-3., -3.], bounds=(-18., 5.)).x
    weights = _ls(basis(c['u'], p), c['v2'])

    def warp(x):
        d = (x - bias) / gain
        return d + ((basis(d @ rotation, p)[:, 1:]) @ weights[1:]) @ rotation

    return warp


def _predict(coords, targets, length, ridge):
    n = len(targets)
    a = coords / coords[:n].std(axis=0).mean()
    train = a[:n]
    k = (1 + a @ train.T / a.shape[1]) ** 2
    k += np.exp(-cdist(a, train, 'sqeuclidean') / (2 * a.shape[1] * length**2))
    return k[n:] @ np.linalg.solve(k[:n] + ridge * np.eye(n), targets)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    y = _predict(_calibration(study)(x), train['y'], 1.2, 3e-5)
    traj = train['traj']
    _, _, vh = np.linalg.svd(traj, full_matrices=False)
    modes = vh[:2]
    state = _predict(x, traj @ modes.T, 1.0, .003) @ modes
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
