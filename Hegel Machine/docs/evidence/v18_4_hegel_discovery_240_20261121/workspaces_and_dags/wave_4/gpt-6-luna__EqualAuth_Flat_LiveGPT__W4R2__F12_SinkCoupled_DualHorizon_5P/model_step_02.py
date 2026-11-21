import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows], dtype=float) - y
    starts = [
        [3.0, 0.5, 1.0, 0.8, 10.0],
        [4.0, 0.3, 0.5, 0.5, 20.0],
        [2.5, 0.7, 2.0, 0.7, 8.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [20.0, 1.0, 100.0, 20.0, 200.0]), max_nfev=20000) for x0 in starts]
    p = min(fits, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return {'params': p}

def predict(row):
    p = _MODEL['params']
    return float(_value(row, p))

_MODEL = {'params': np.array([3.0, 0.5, 1.0, 0.8, 10.0])}
def _bound_predict(row):
    return predict(row)
