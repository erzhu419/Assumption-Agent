import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u_n = max(1e-8, float(row['u_now']) / 0.05)
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.8, 1.0, 0.5, 0.2]), bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 100.0, 0.999, 10.0]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(train_rows)
    return float(_predict_params(p, row))