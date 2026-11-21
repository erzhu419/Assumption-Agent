import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.5, np.inf]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(row, fit(row['train_rows'])))