"""Panel-calibrated spatial regression with a shared temporal response."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _affine(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slopes = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)
    offsets = np.median(v - u * slopes, axis=0)
    for _ in range(3):
        r = np.linalg.norm(v - u * slopes - offsets, axis=1)
        keep = r <= max(6 * np.median(r), .0005)
        if keep.sum() >= 6:
            for k in range(u.shape[1]):
                offsets[k], slopes[k] = np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep, k]], v[keep, k], rcond=None)[0]
    return slopes, offsets


def _coordinates(study, x):
    a, b = _affine(study['panel_a'])
    z = (x - b) / a
    p = study['panel_b']
    u = np.asarray(p['u'])
    w = np.linalg.lstsq(np.c_[np.ones(len(u)), u], p['v'], rcond=None)[0]
    return z, np.c_[np.ones(len(z)), z] @ w


def _kernel_predict(x, target, width, ridge, mode='radial'):
    n = len(target)
    d = cdist(x, x[:n], 'sqeuclidean')
    r = np.exp(-width * d)
    if mode == 'multiscale':
        r = .6 * r + .4 * np.exp(-.22 * width * d)
    elif mode == 'triple':
        delta = (x[:, None, :] - x[None, :n, :]) ** 2
        r = .2 * r + .8 * np.exp(-width * (d[:, :, None] - delta)).mean(axis=2)
    k = 1. + x @ x[:n].T + r
    fact = cho_factor(k[:n] + ridge * np.eye(n), lower=True)
    return k[n:] @ cho_solve(fact, target)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate([train['x'], study['test']['x']], axis=0)
    z, bz = _coordinates(study, x)
    b = study['panel_b']
    u = np.asarray(b['u'])
    w = np.linalg.lstsq(np.c_[np.ones(len(u)), u], b['v'], rcond=None)[0]
    bx = x @ w[1:] + w[0]

    # Use a warped calibrated metric and a smoother multiscale chart.
    target = np.asarray(train['y'])
    y = .65 * _kernel_predict(np.sinh(.75 * z) / .75, target, .30, .006)
    y += .35 * _kernel_predict(bz, target, .45, .0015, 'multiscale')

    # Fit the shared time profile on training trajectories only.
    trajectories = np.asarray(train['traj'])
    h = np.linalg.svd(trajectories, full_matrices=False)[2][0]
    amplitude = trajectories @ h
    a = .65 * _kernel_predict(np.sinh(.35 * x) / .35, amplitude, .15, .0002)
    a += .35 * _kernel_predict(np.sin(.35 * bx) / .35, amplitude, .30, .0002, 'triple')
    state = a[:, None] * h
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
