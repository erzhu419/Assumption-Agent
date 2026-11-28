"""Robust diagonal calibration and profiled spatial/temporal readouts."""
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares, minimize


def _coordinates(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    du, dv = u[i] - u[j], v[i] - v[j]
    gain = np.array([np.median(dv[abs(du[:, k]) > .1, k] / du[abs(du[:, k]) > .1, k]) for k in range(u.shape[1])])
    offset = np.median(v - u * gain, axis=0)
    for _ in range(3):
        error = np.linalg.norm(v - offset - gain * u, axis=1)
        med = np.median(error)
        keep = error <= max(5 * med, 1e-10)
        uc, vc = u[keep], v[keep]
        um, vm = uc.mean(axis=0), vc.mean(axis=0)
        gain = np.sum((uc - um) * (vc - vm), axis=0) / np.sum((uc - um)**2, axis=0)
        offset = vm - um * gain
    return (x - offset) / gain


def _quadratic(u):
    return np.column_stack([np.ones(len(u)), u, u*u] + [u[:, i]*u[:, j] for i, j in combinations(range(u.shape[1]), 2)])


def _design(u, base, p):
    c, k, r = np.exp(p)
    w = abs(u)
    h = np.sign(u) * np.exp(k*np.log(np.maximum(w, 1e-30)) - r*np.log1p(c*w*w))
    return np.column_stack((base, h))


def _profile(u, target):
    base = _quadratic(u)

    def objective(p):
        g = _design(u, base, p)
        co = np.linalg.lstsq(g, target, rcond=None)[0]
        return np.mean((g @ co - target)**2)

    starts = [[.6, 3, 1], [2, 1, 3], [.5, 6, 2], [10, 10, 4.5]]
    return min((minimize(objective, np.log(p), method='L-BFGS-B',
                         bounds=[(-5, 4), (-1.5, 2.45), (-3, 3.5)],
                         options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 400})
                for p in starts), key=lambda opt: opt.fun).x


def _time_basis(times, z):
    def basis(p):
        return -np.expm1(-times[:, None] * np.exp(p))

    def residual(p):
        f = basis(p)
        return (f @ np.linalg.lstsq(f, z.T, rcond=None)[0] - z.T).ravel()

    fit = least_squares(residual, np.log([.25, 1.5]), bounds=(np.log(.04), np.log(10)))
    return np.linalg.qr(basis(fit.x), mode='reduced')[0]


def _predict(study, temporal='exponential'):
    tr = study['train']
    x = np.asarray(tr['x'], dtype=float)
    xt = np.asarray(study['test']['x'], dtype=float)
    u, ut = np.split(_coordinates(study['panel_a'], np.concatenate((x, xt))), [len(x)])
    z = np.asarray(tr['traj'], dtype=float)
    y = np.asarray(tr['y'], dtype=float)
    if temporal == 'exponential':
        time = _time_basis(np.asarray(study['eval_times'], dtype=float), z)
    elif temporal == 'svd':
        time = np.linalg.svd(z, full_matrices=False)[2][:2].T
    else:
        time = np.eye(z.shape[1])
    p = _profile(u, np.column_stack((y, z @ time)))
    base, base_test = _quadratic(u), _quadratic(ut)
    g, gt = _design(u, base, p), _design(ut, base_test, p)
    # Reuse the learned response coordinates in the scalar interaction terms.
    n_base = base.shape[1]
    h, ht = g[:, n_base:], gt[:, n_base:]
    scale = np.maximum(np.max(np.abs(h), axis=0), np.finfo(float).tiny)
    h, ht = h / scale, ht / scale
    pairs = list(combinations(range(u.shape[1]), 2))
    scalar_train = np.column_stack([g[:, :n_base], h] + [h[:, i] * h[:, j] for i, j in pairs])
    scalar_test = np.column_stack([gt[:, :n_base], ht] + [ht[:, i] * ht[:, j] for i, j in pairs])
    yp = scalar_test @ np.linalg.lstsq(scalar_train, y, rcond=None)[0]
    zp = (gt @ np.linalg.lstsq(g, z @ time, rcond=None)[0]) @ time.T
    return yp, zp


def solve_batch(study: dict) -> dict:
    """Predict the scalar readout and state at every requested evaluation time."""
    y, z = _predict(study)
    return {'y_hat': y, 'state_pred': z,
            'diagnostics_vec': np.array([y.mean(), y.std(), z[:, -1].mean(), z[:, -1].std()])}
