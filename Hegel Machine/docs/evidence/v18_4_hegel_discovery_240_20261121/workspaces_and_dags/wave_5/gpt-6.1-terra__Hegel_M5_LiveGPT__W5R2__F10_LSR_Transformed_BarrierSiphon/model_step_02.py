import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) / 0.02 for r in train_rows])
    s = np.asarray([float(r['s_grad']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u) - y
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(residual, [4.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 30.0, 1.0, 20.0]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    j0, beta, rho, c = _params
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return float(j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u))
