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
        result = least_squares(residual, [4.0, beta, 0.7, 0.0], bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]), max_nfev=10000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    j, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c*math.sqrt(u))
