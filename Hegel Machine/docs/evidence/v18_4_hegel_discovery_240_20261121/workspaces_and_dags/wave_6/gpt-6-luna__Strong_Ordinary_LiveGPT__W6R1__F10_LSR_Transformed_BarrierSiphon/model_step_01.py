import math
import numpy as np
from scipy.optimize import least_squares

def _predict_with(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    def residual(p):
        return np.array([_predict_with(p, r) - float(r['y']) for r in train_rows])
    best = None
    for p0 in ([5.0, 1.0, 0.5, 0.0], [10.0, 2.0, 0.7, 0.2], [2.0, 0.2, 0.3, -0.2]):
        result = least_squares(residual, p0, bounds=([0.0, 0.0, -5.0, -20.0], [1000.0, 100.0, 5.0, 20.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict requires fitted parameters via fit(train_rows); use predict_with_params')
    return float(_predict_with(p, row))