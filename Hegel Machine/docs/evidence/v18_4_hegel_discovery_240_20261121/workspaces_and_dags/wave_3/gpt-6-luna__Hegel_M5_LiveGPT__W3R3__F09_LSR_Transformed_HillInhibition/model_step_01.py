import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.1

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residuals(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residuals, x0=np.array([3.0, 1.0, 0.1, 0.01]), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
    return {'params': result.x.tolist()}

def predict(row):
    params = row.get('_fit_params')
    if params is None:
        raise ValueError('predict requires row with _fit_params returned by fit, or parameters supplied by caller')
    return float(_predict_params(row, params))