import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [1.0, 0.1, 0.7, 0.0],
        [5.0, 1.0, 0.5, 0.0],
        [10.0, 3.0, 0.9, -0.5],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.sum(result.fun ** 2)))
    return best.x.tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(row, p))