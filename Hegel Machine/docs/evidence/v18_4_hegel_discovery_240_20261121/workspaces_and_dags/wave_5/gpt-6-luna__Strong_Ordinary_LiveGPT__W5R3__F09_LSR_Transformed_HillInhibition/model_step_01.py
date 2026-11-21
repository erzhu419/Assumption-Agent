import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u_scale = 0.1
    def predict_params(p, row):
        vmax, k_half, alpha_inh, c_leak = p
        u = row['u_now'] / u_scale
        s = row['s_grad']
        return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u
    def residuals(p):
        return np.array([predict_params(p, r) - r['y'] for r in rows])
    result = least_squares(residuals, x0=np.array([4.0, 1.0, 0.5, 0.05]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
    params = result.x
    def predict(row):
        return float(predict_params(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')