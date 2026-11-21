import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02

def _features(row):
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return u_n, s

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n, s = _features(row)
    return (j0 * u_n * math.exp(-beta_bar / (0.5 + u_n))
            * (1.0 - rho * s / (1.0 + s))
            + c_sqrt * math.sqrt(max(u_n, 0.0)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(
        residual,
        x0=np.array([1.0, 1.0, 0.7, 0.1]),
        bounds=(np.array([0.0, 0.0, 0.0, -np.inf]),
                np.array([np.inf, np.inf, 0.95, np.inf])),
        max_nfev=20000
    )
    return result.x

def predict(row):
    return float(_value(row, fit([dict(row, y=0.0)])))