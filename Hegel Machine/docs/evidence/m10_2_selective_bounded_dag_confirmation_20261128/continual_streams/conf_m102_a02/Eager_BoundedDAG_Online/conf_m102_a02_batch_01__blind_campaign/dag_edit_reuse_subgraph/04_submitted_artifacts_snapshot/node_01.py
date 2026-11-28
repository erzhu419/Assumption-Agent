"""Fit readouts in robustly calibrated input coordinates."""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar


def _calibrate(panel):
    u, v = np.asarray(panel['u'], float), np.asarray(panel['v'], float)
    offsets, gains = [], []
    i, j = np.triu_indices(len(u), 1)
    for a, b in zip(u.T, v.T):
        delta = a[j] - a[i]
        good = np.abs(delta) > 1e-10
        if not np.any(good):
            offsets.append(0.)
            gains.append(1.)
            continue
        m = np.median((b[j] - b[i])[good] / delta[good])
        c = np.median(b - m*a)
        r = np.abs(b - c - m*a)
        keep = r <= max(1e-8, 5*np.median(r))
        if np.count_nonzero(keep) >= 2:
            c, m = np.linalg.lstsq(np.column_stack((np.ones(sum(keep)), a[keep])), b[keep], rcond=None)[0]
        offsets.append(c)
        gains.append(m if abs(m) > 1e-10 else 1.)
    return np.array(offsets), np.array(gains)


def _features(z, frequency):
    w = np.sin(frequency * z)**3
    pairs = [w[:, i]*w[:, j] for i, j in combinations(range(z.shape[1]), 2)]
    return np.column_stack((np.ones(len(z)), w, z, w*w, *pairs))


def _fit(z, targets, square_penalty, pair_penalty, rank=None):
    n = z.shape[1]
    scale = np.maximum(targets.std(axis=0), 1e-8)

    def profile(frequency):
        design = _features(z, frequency)
        penalty = design.std(axis=0)**2
        penalty[1:1+2*n] *= .0135
        penalty[1+2*n:1+3*n] *= square_penalty
        penalty[1+3*n:] *= pair_penalty
        coef = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ targets)
        return design, coef

    def loss(frequency):
        design, coef = profile(frequency)
        return np.mean(((design @ coef - targets) / scale)**2)

    frequency = minimize_scalar(loss, bounds=(.5, 1.2), method='bounded',
                                options={'xatol': 1e-5}).x
    design, coef = profile(frequency)
    if rank is not None:
        _, _, vt = np.linalg.svd(design @ coef, full_matrices=False)
        modes = vt[:rank].T
        coef = (coef @ modes) @ modes.T
    return frequency, coef


def run_node(inputs: dict) -> dict:
    study, frozen = inputs['__input__'], inputs['node_00']
    offsets, gains = _calibrate(study['panel_a'])
    z = (np.asarray(frozen['a'], float)[:, 1:] - offsets) / gains
    y = np.asarray(study['train']['y'], float).reshape(-1, 1)
    states = np.asarray(study['train']['traj'], float)
    fy, by = _fit(z, y, 1.4, 6.)
    fs, bs = _fit(z, states, 5.6, 20., rank=2)
    return {'offsets': offsets, 'gains': gains, 'frequencies': np.array([fy, fs]),
            'endpoint_coef': by, 'trajectory_coef': bs}
