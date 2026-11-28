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


def _predict(charts, targets):
    n = len(targets)
    yw = targets / np.maximum(targets.std(axis=0), 1e-12)
    eye = np.eye(n)
    estimates, scores, groups = [], [], []
    for j, X in enumerate(charts):
        for h in [0., .22, .45]:
            Z = X if h == 0 else np.tanh(h * X) / h
            L = Z @ Z[:n].T
            D = cdist(Z, Z[:n], 'sqeuclidean')
            for scale in [.12, .22, .34, .48, .65, .85]:
                for q in [.02, .12, .45]:
                    for mode in (0, 1):
                        R = np.exp(-scale * D)
                        if mode:
                            A = (Z[:, None, :] - Z[None, :n, :]) ** 2
                            R = .6 * R + .4 * np.exp(-2 * scale * A).mean(axis=2)
                        K = 1. + L + q * L**2 + R
                        for ridge in [.00012, .00065, .0025, .009]:
                            f = cho_factor(K[:n] + ridge * eye, lower=True, check_finite=False)
                            alpha = cho_solve(f, yw, check_finite=False)
                            diagonal = cho_solve(f, eye, check_finite=False).diagonal()
                            err = alpha / diagonal[:, None]
                            scores.append(.68 * np.mean(abs(err), axis=0) + .32 * np.sqrt(np.mean(err**2, axis=0)))
                            estimates.append(K[n:] @ alpha)
                            groups.append((j,h,scale,q,mode,ridge))
    scores = np.array(scores)
    estimates = np.array(estimates)
    output = []
    for k in range(yw.shape[1]):
        s = scores[:, k]; w = np.exp(-(s / s.min() - 1) / .065); w /= w.sum()
        output.append(w @ estimates[:, :, k] * max(targets.std(axis=0)[k], 1e-12))
    return np.array(output).T


def solve_batch(study):
    tr = study['train']
    x = np.concatenate([tr['x'], study['test']['x']], axis=0)
    charts = _coordinates(study, x)
    T = np.asarray(tr['traj'])
    h = np.linalg.svd(T, full_matrices=False)[2][0]
    p = _predict(charts, np.c_[tr['y'], T @ h])
    y = p[:, 0]
    state = p[:, 1, None] * h
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
