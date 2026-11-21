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
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.70, 0.5]), bounds=([0.0, 0.0, 0.05, 0.0], [100.0, 20.0, 0.95, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(row, p))