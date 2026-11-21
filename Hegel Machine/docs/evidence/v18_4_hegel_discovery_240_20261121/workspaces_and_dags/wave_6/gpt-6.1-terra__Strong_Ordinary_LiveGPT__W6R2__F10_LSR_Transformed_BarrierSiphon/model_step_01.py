import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_evaluate(p, r) - float(r['y']) for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 7.0):
        for rho in (0.3, 0.7, 1.0):
            result = least_squares(residual, [2.0, beta, rho, 0.1], bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 30.0, 2.0, 20.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, row))