import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    p0 = np.array([4.0, 0.5, 0.5, 0.5, 12.0])
    lower = np.array([0.0, 0.0, 1e-4, 0.0, 1e-3])
    upper = np.array([30.0, 1.0, 30.0, 10.0, 200.0])
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_predict_params(row, p))