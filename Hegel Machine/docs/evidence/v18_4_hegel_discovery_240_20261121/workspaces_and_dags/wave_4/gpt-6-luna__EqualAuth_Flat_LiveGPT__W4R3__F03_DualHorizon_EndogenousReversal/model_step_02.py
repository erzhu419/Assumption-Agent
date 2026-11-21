import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    amp_fast, amp_slow, log_tau_fast, log_tau_gap = p
    tau_fast = math.exp(log_tau_fast)
    tau_slow = tau_fast + math.exp(log_tau_gap)
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    p0 = np.array([1.0, 1.0, math.log(0.4), math.log(8.0)], dtype=float)
    lower = np.array([0.0, 0.0, math.log(1e-3), math.log(1e-3)], dtype=float)
    upper = np.array([20.0, 20.0, math.log(100.0), math.log(1000.0)], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000, x_scale='jac')
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_predict_params(row, p))