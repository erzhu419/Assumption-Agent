import math
import numpy as np
from scipy.optimize import least_squares

def _value(p, row):
    vmax, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    h = float(row['H'])
    s = float(row.get('s_grad') or 0.0)
    return (vmax * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    result = least_squares(residual, x0=[4.0, 0.7, 1.2, 20.0],
                           bounds=([0.0, 1e-5, 0.0, 1e-5], [100.0, 100.0, 100.0, 1000.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(fit(_TRAIN_ROWS), row))