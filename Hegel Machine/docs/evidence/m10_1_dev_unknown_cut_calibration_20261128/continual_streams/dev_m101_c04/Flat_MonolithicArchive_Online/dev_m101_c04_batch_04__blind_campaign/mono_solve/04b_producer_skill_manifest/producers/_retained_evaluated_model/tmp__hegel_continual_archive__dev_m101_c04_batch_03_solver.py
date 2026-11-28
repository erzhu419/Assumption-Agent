"""Calibrated dead-zone operator and two-mode relaxation predictor."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    params = []
    for x, y in zip(u.T, v.T):
        dx = x[i] - x[j]
        valid = dx != 0
        gain = np.median((y[i] - y[j])[valid] / dx[valid])
        offset = np.median(y - gain*x)
        error = abs(y - offset - gain*x)
        keep = error <= max(.001, 5*np.median(error))
        params.append(_ls(_affine(x[keep]), y[keep]))
    return np.asarray(params).T


def _operator(panel):
    u, v = panel['u'], panel['v']
    vmean = v.mean(axis=0)
    vc = v - vmean

    def gate(x, p):
        return np.sign(x) * np.maximum(abs(x) - p[0], 0)**p[1]

    def profile(p, predict=False):
        f = gate(u, p)
        center = f.mean(axis=0)
        f = f - center
        left, sing, right = np.linalg.svd(f.T @ vc)
        mix = (left @ right) * (sing.sum() / np.sum(f*f))
        if predict:
            return lambda x: (gate(x, p) - center) @ mix + vmean
        return (f @ mix - vc).ravel()

    result = least_squares(profile, [.35, 1.5], bounds=([0, .6], [.8, 4]))
    return profile(result.x, True)


def _features(study, x):
    offset, gain = _calibrate(study['panel_a'])
    z = _operator(study['panel_b'])((x-offset)/gain)
    cross = z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]
    return _affine(z), cross


def _time_basis(t, traj):
    def basis(p):
        return -np.expm1(-t[:, None]*np.exp(p))

    def residual(p):
        h = basis(p)
        return (h @ _ls(h, traj.T) - traj.T).ravel()

    p = least_squares(residual, np.log([.4, 1.8]), bounds=(np.log(.03), np.log(15))).x
    return basis(p)


def solve_batch(study):
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    f, cross = _features(study, x)
    n = len(train['x'])
    fy = np.column_stack((f, cross))
    y = fy[n:] @ _ls(fy[:n], train['y'])
    h = _time_basis(study['eval_times'], train['traj'])
    q = _ls(h, train['traj'].T).T
    state = (f[n:] @ _ls(f[:n], q)) @ h.T
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
