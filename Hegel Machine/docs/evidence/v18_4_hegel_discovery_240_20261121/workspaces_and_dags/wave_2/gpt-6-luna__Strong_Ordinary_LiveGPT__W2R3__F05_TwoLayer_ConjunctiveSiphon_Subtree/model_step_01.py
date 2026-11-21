import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.05

def _predict(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict(r, p) for r in rows]) - y
    best = None
    for start in ([2.0, 0.5, 0.0], [3.0, 0.7, 0.1], [1.0, 0.3, -0.1]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit([row])
    return float(_predict(row, p))