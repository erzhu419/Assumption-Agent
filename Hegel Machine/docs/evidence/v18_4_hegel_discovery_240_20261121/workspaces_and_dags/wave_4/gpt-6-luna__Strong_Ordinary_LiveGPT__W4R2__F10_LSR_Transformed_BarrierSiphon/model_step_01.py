import math
import numpy as np
from scipy.optimize import least_squares

def _value(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    flux = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    best = None
    for start in ([2.0, 1.0, 0.5, 0.0], [1.0, 0.2, 0.2, 0.5], [4.0, 2.0, 0.8, -0.5], [1.0, 3.0, 0.6, 1.0]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return {'params': best.x.tolist()}

def predict(row):
    return _value(np.asarray(fit_params, dtype=float), row)