import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.70, 0.2]), bounds=([0.0, 0.0, 0.05, 0.0], [100.0, 20.0, 0.95, 100.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_predict_params(row, p))