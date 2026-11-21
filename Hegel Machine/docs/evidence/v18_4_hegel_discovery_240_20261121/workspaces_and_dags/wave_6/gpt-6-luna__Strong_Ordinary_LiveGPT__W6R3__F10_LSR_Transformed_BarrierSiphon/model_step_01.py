import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for start in ([3.0, 0.5, 0.5, 0.1], [5.0, 1.0, 0.7, 0.5], [2.0, 0.1, 0.2, -0.2]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 20.0, 1.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict expects fitted parameters in row["params"]')
    return float(_value(row, p))