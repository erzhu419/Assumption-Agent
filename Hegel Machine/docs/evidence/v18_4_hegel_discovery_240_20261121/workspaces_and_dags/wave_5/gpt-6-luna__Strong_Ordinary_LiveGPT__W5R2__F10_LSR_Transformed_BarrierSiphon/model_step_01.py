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
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = ([1.0, 1.0, 0.5, 0.0], [2.0, 3.0, 0.7, 0.1], [0.5, 0.2, 0.2, 0.5])
    best = None
    for p0 in starts:
        result = least_squares(residual, np.asarray(p0, dtype=float), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else row['_fitted_parameters']
    return float(_value(row, p))