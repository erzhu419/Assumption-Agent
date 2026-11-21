import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    starts = [
        [5.0, 0.7, 0.0, 1.0, 2.6],
        [8.0, 0.5, -0.5, 1.5, 2.5],
        [3.0, 0.9, 0.5, 0.8, 2.7],
    ]
    best = None
    for x0 in starts:
        res = least_squares(
            lambda p: np.array([_value(r, p) for r in train_rows]) - y,
            x0=x0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.5, 10.0, 20.0, 3.2]),
            max_nfev=20000,
        )
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    return best.x

def predict(row):
    return float(_value(row, fit_params))