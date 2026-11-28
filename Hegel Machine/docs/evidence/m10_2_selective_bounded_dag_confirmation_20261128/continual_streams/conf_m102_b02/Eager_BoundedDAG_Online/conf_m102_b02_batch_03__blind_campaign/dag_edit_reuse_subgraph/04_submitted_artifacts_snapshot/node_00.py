"""Recover latent coordinates from two paired calibration panels."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _powers(d):
    return [()] + [(i,) for i in range(d)] + [p for k in (2, 3) for p in combinations_with_replacement(range(d), k)]


def _design(x, powers):
    return np.stack([np.prod(x[..., p], axis=-1) if p else np.ones(x.shape[:-1]) for p in powers], axis=-1)


def _calibrate(u, v):
    powers = _powers(u.shape[1])
    a = _design(u, powers[:1+u.shape[1]])
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        w = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
        r = np.linalg.norm(a @ w - v, axis=1)
        proposed = r < max(0.003, 4*np.median(r[keep]))
        if proposed.sum() < len(powers) or np.array_equal(proposed, keep):
            break
        keep = proposed
    a = _design(u[keep], powers)
    w = np.linalg.lstsq(a, v[keep], rcond=None)[0]
    # Refit after dropping coefficients compatible with reference noise.
    mask = np.max(np.abs(w), axis=1) > 0.002
    mask[:1+u.shape[1]] = True
    return [p for p, m in zip(powers, mask) if m], np.linalg.lstsq(a[:, mask], v[keep], rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    powers, w = _calibrate(np.asarray(a['u'], float), np.asarray(a['v'], float))
    x, xt = (np.asarray(study[k]['x'], float) for k in ('train', 'test'))
    observations = np.concatenate((x, xt))
    d = x.shape[1]
    initial = np.linalg.solve(w[1:d+1].T, (observations - w[0]).T).T
    latent = np.empty_like(observations)
    for i, (v, u0) in enumerate(zip(observations, initial)):
        latent[i] = least_squares(lambda u: _design(u, powers) @ w - v, u0, gtol=1e-12, max_nfev=60).x
    ap = _powers(d)[:d+1]
    wb = np.linalg.lstsq(_design(np.asarray(b['u'], float), ap), np.asarray(b['v'], float), rcond=None)[0]
    z = _design(latent, ap) @ wb
    return {'z': z[:len(x)].copy(), 'zt': z[len(x):].copy()}
