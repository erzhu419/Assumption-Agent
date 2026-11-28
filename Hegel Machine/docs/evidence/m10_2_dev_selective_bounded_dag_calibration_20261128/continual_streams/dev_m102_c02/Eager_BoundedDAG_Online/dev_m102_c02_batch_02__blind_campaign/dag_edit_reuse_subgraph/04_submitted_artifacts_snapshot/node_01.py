"""Fit the scalar readout and pass the affine trajectory projection downstream."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _interaction(x):
    return np.sum(x[:, :2] * x[:, 2:], axis=1)


def _readout(x, interaction, params):
    z = x @ params[1:5]
    scale, weight = params[6:8]
    return params[0] + params[5] * interaction + (1.0 - weight) * z + weight * np.arctan(scale * z) / scale


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    coords = inputs['node_00']
    c = np.asarray(coords['c'], dtype=float)
    query = np.asarray(coords['query'], dtype=float)
    train = study['train']
    y = np.asarray(train['y'], dtype=float)
    design = _affine(c)
    new_design = _affine(query)
    interaction = _interaction(c)
    initial = np.linalg.lstsq(np.column_stack((design, interaction)), y, rcond=None)[0]
    params = least_squares(
        lambda p: _readout(c, interaction, p) - y,
        np.r_[initial, 0.35, 1.0],
        bounds=(np.r_[np.full(6, -np.inf), 0.015, 0.0],
                np.r_[np.full(6, np.inf), 1.5, 5.0]),
        max_nfev=200,
    ).x
    beta = np.linalg.lstsq(design, np.asarray(train['traj'], dtype=float), rcond=None)[0]
    times = np.asarray(study['eval_times'], dtype=float)
    t = times / max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    return {'y_hat': _readout(query, _interaction(query), params),
            'fitted': design @ beta, 'beta': beta, 'new_design': new_design, 't': t}
