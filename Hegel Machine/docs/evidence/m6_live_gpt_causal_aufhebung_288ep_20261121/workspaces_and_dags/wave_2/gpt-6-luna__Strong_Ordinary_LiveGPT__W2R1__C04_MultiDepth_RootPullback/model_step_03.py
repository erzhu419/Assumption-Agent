import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    dose = outdoor + indoor
    loss_basis = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return dose, loss_basis

def _predict_params(row, params):
    g_root, k_e, c_imp = params
    dose, loss_basis = _features(row)
    return g_root * dose / (k_e + dose) - c_imp * loss_basis

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(params):
        return np.asarray([_predict_params(r, params) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([8.0, 0.5, 1.0]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))