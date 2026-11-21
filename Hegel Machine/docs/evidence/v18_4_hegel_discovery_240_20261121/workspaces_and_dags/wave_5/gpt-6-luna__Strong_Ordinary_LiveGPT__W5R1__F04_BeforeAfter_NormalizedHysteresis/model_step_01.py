import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004

def _predict_params(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.0]), bounds=([-1000.0, 1e-6, -1000.0], [1000.0, 100.0, 1000.0]))
    return {'params': result.x}

def predict(row):
    return float(_predict_params(row, fit_params))
