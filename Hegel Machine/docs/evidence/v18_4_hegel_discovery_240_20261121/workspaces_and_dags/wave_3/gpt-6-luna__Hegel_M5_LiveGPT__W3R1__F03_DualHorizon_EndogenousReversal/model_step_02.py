import math
import numpy as np
from scipy.optimize import least_squares

def _value(p, row):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(p, r) for r in rows]) - y
    result = least_squares(residual, x0=[1.2, 0.5, 0.8, 8.0], bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 100.0, 20.0, 200.0]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(p, row))