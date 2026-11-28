"""Calibrated dead-zone operator with a rational scalar readout."""
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


def _scalar(z, y, nz):
    interaction = z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]
    x = _affine(np.column_stack((z, interaction)))
    train = x[:nz]
    init = _ls(train, y)

    def design(w):
        return np.column_stack((np.ones(len(w)), w, w/(1+w*w)))

    def residual(p):
        a = design(train @ p)
        return a @ _ls(a, y) - y

    best = min((least_squares(residual, scale*init, max_nfev=140)
                for scale in (.55, 1.2, 2.3)), key=lambda fit: fit.cost)
    a = design(x @ best.x)
    return a[nz:] @ _ls(a[:nz], y)


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    offset, gain = _calibrate(study['panel_a'])
    z = _operator(study['panel_b'])((x-offset)/gain)
    y = _scalar(z, train['y'], n)
    f = _affine(z)
    state = f[n:] @ _ls(f[:n], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
