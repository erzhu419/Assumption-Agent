import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        j, beta, rho, c = p
        return j*u*np.exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c*np.sqrt(u) - y
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        for j in (3.0, 8.0):
            result = least_squares(residual, [j, beta, 0.7, 0.3], bounds=([0, 0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    j, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c*math.sqrt(u))