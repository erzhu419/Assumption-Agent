from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    design = _affine(u)
    residual = np.linalg.norm(v - design @ _fit(design, v), axis=1)
    keep = residual < 3.0
    if keep.sum() < max(5, len(u) // 2):
        keep = residual <= np.median(residual)
    coeff = np.stack([_fit(design[keep][:, [0, j + 1]], v[keep, j])
                      for j in range(u.shape[1])], axis=1)
    return (x - coeff[0]) / coeff[1]


def _odd(x, scale):
    return x / (1.0 + scale * x * x)


def _even(x, scale):
    return np.log1p(scale * x * x)


def _panel_map(u, v, x, basis):
    affine_u = _affine(u)

    def design(a):
        return np.column_stack((affine_u, basis(u, a)))

    def residual(a):
        d = design(a)
        return (d @ _fit(d, v) - v).ravel()

    scale = least_squares(residual, np.full(u.shape[1], 0.6),
                          bounds=(0.01, 5.0), gtol=1e-12).x
    coeff = _fit(design(scale), v)
    return np.column_stack((_affine(x), basis(x, scale))) @ coeff


def _predict(study):
    train = study['train']
    n = len(train['x'])
    x = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    panel = study['panel_c']
    v = np.column_stack((_panel_map(panel['u'], panel['v1'], x, _odd),
                         _panel_map(panel['u'], panel['v2'], x, _even)))
    features = np.column_stack([v] + [v[:, i] * v[:, j]
                               for i, j in combinations_with_replacement(range(v.shape[1]), 2)])
    features = (features - features[:n].mean(axis=0)) / np.maximum(1e-8, features[:n].std(axis=0))
    labels = np.column_stack((train['y'], train['traj']))
    mean = labels.mean(axis=0)
    f = features[:n]
    weights = np.linalg.solve(f @ f.T + 0.003 * np.eye(n), labels - mean)
    return features[n:] @ f.T @ weights + mean


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    prediction = _predict(study)
    # Express the calibrated readout in the coordinates of the frozen blend.
    y = (prediction[:, 0] - 0.5 * sparse['y_hat']) / 0.5
    state = (prediction[:, 1:] - 0.35 * sparse['state_pred']) / 0.65
    return {'panel_pred': np.column_stack((y, state))}
