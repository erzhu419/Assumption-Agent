from itertools import combinations
import numpy as np
from scipy.optimize import least_squares

_PAIRS = np.array(list(combinations(range(4), 2)))
_LEFT, _RIGHT = _PAIRS.T

def _bridge_basis(load):
    return np.column_stack((np.ones(len(load)), load, load[:, _LEFT] * load[:, _RIGHT], load**3))

def _fit_bridge(cal):
    load = np.asarray(cal['true_channel_loads'], float)
    raw = np.asarray(cal['bridge_raw'], float)
    design = _bridge_basis(load)
    start = np.linalg.lstsq(design, raw, rcond=None)[0]
    robust = least_squares(lambda b: (design @ b.reshape(start.shape) - raw).ravel(),
                           start.ravel(), jac=lambda b: np.kron(design, np.eye(4)),
                           loss='soft_l1', f_scale=0.003).x.reshape(start.shape)
    residual = np.linalg.norm(design @ robust - raw, axis=1)
    inlier = residual < max(0.003, 8 * np.median(residual))
    return np.linalg.lstsq(design[inlier], raw[inlier], rcond=None)[0]

def _decode_bridge(raw, coef):
    raw = np.asarray(raw, float)
    shape = raw.shape
    target = raw.reshape(-1, 4)
    load = (target - coef[0]) @ np.linalg.inv(coef[1:5])
    for _ in range(20):
        residual = _bridge_basis(load) @ coef - target
        if np.max(np.abs(residual), initial=0.0) < 1e-11:
            break
        jac = coef[1:5] + 3 * load[:, :, None]**2 * coef[11:15]
        for k, (i, j) in enumerate(_PAIRS):
            jac[:, i] += load[:, j, None] * coef[5+k]
            jac[:, j] += load[:, i, None] * coef[5+k]
        load -= np.linalg.solve(jac.transpose(0, 2, 1), residual[:, :, None])[:, :, 0]
    return load.reshape(shape)
