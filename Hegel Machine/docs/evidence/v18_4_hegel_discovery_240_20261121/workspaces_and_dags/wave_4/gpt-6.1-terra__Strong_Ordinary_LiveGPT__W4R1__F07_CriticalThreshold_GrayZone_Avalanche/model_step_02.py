import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, rho, c, k, crit = p
        return vmax * (1 - rho*s/(1+s)) * u/(1.2+u) + c*np.sqrt(u) - k*np.maximum(0, s-crit)*u - y
    best = None
    for crit in (2.50, 2.59, 2.68):
        result = least_squares(residual, [6.0, 0.7, 1.0, 2.0, crit], bounds=([0, 0, -np.inf, 0, 2.50], [np.inf, 1, np.inf, np.inf, 2.68]), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    vmax, rho, c, k, crit = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(vmax*(1-rho*s/(1+s))*u/(1.2+u) + c*np.sqrt(u) - k*max(0.0, s-crit)*u)
