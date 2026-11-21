import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.3, 0.3, 0.5, 15.0]), bounds=([0.0, 0.0, 0.0, 1e-4, 0.5], [100.0, 1.0, 10.0, 10.0, 200.0]), max_nfev=20000)
    return result.x

def predict(row):
    raise RuntimeError('predict(row) requires fitted parameters; call fit(train_rows) and pass the returned parameters to predict_with_params(row, params)')

def predict_with_params(row, params):
    return float(_value(row, params))