import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    un = float(row['u_now']) / 3.0
    sg = float(row['s_grad'])
    return un, sg

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        j0, beta, rho, c_sqrt = p
        pred = []
        for r in rows:
            un, sg = _features(r)
            pred.append(j0 * un * math.exp(-beta / (0.5 + un)) * (1.0 - rho * sg / (1.0 + sg)) + c_sqrt * math.sqrt(max(un, 0.0)))
        return np.asarray(pred) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.3, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 50.0, 1.0, np.inf]), max_nfev=20000)
    return result.x

def predict(row):
    j0, beta, rho, c_sqrt = fit([row])
    un, sg = _features(row)
    return float(j0 * un * math.exp(-beta / (0.5 + un)) * (1.0 - rho * sg / (1.0 + sg)) + c_sqrt * math.sqrt(max(un, 0.0)))