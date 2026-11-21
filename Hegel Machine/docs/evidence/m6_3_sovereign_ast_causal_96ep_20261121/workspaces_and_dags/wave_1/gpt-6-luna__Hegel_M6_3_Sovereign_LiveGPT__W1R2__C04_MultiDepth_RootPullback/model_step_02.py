import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    e_indoor = float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    e_eff = float(row['u_outdoor']) * float(row['p_weather']) + e_indoor
    loss_basis = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return e_eff, loss_basis

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    feats = [_features(r) for r in train_rows]
    def residuals(params):
        g_root, k_e, c_imp = params
        return np.asarray([g_root * e / (k_e + e) - c_imp * loss for e, loss in feats]) - ys
    result = least_squares(residuals, x0=np.array([7.0, 1.2, 1.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return predict

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    e_eff, loss_basis = _features(row)
    g_root, k_e, c_imp = _PARAMS
    return float(g_root * e_eff / (k_e + e_eff) - c_imp * loss_basis)
