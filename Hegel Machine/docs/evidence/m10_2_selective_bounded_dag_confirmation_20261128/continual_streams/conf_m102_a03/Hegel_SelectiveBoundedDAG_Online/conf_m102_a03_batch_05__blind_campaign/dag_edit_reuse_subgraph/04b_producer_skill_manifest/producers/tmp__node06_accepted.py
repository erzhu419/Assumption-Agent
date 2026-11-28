"""Calibrate the sensor coordinates and construct the scalar and state kernels."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _sensor_calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = np.stack((np.ones_like(u), u), axis=-1)
    w = np.ones(len(u))
    for _ in range(24):
        co = np.column_stack([_fit(a[:, j] * w[:, None], v[:, j] * w)
                              for j in range(u.shape[1])])
        r = np.linalg.norm(v - co[0] - co[1] * u, axis=1)
        w = np.sqrt(np.minimum(1., 1.5 * max(np.median(r), 1e-12) / np.maximum(r, 1e-12)))
    return co


def _readout_coordinates(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    d = u.shape[1]

    def basis(x, p, trig):
        h = x * np.exp(p[:d]) + p[d:]
        if trig:
            return np.column_stack((_affine(x), x*x, np.sin(h)))
        return np.column_stack((_affine(x), np.tanh(h)))

    candidates = []
    for trig in (False, True):
        def residual(p):
            f = basis(u, p, trig)
            return (f @ _fit(f, v) - v).ravel()

        result = min((least_squares(residual, np.r_[np.full(d, k), np.zeros(d)],
                                    bounds=(-2., 2.), max_nfev=90)
                      for k in (-.3, .4, 1.)), key=lambda r: r.cost)
        f = basis(u, result.x, trig)
        score = np.linalg.norm(result.fun) * (1 + f.shape[1] / 20.)
        candidates.append((score, basis(z, result.x, trig) @ _fit(f, v)))
    return min(candidates, key=lambda item: item[0])[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    co = _sensor_calibration(study['panel_a'])
    x, xt = (np.asarray(block['x'], dtype=float) for block in (train, study['test']))
    n = len(x)
    z = (np.concatenate((x, xt), axis=0) - co[0]) / co[1]
    w = _readout_coordinates(study['panel_b'], z)
    w = (w - w[:n].mean(0)) / np.maximum(w[:n].std(0), 1e-10)
    wt, w = w[n:], w[:n]

    def scalar(a, b):
        d = a.shape[1]
        return 1. + a @ b.T / d + np.exp(-.5 * cdist(a, b, 'sqeuclidean') / d)

    def state(a, b):
        return 1. + a @ b.T + np.exp(-1.2 * (a[:, None, :] - b[None, :, :])**2).mean(2)

    modes = np.linalg.svd(np.asarray(train['traj'], dtype=float), full_matrices=False)[2][:2]
    ks = state(z[:n], z[:n]) + .001 * np.eye(n)
    kt = state(z[n:], z[:n])
    return {'readout_train': scalar(w, w) + .01 * np.eye(n),
            'readout_test': scalar(wt, w),
            'state_train': np.stack([ks] * len(modes)),
            'modes': modes,
            'zt': np.concatenate([kt] * len(modes), axis=1)}
