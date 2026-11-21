import math
import numpy as np
from scipy.optimize import least_squares

def _basis(row, p):
    j0, beta, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    factor = 1.0 - rho * s / (1.0 + s)
    return j0 * u * math.exp(-beta / (0.5 + u)) * factor + c_sqrt * math.sqrt(u)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_basis(r, p) for r in train_rows]) - ys
    starts = [[4.0, 0.5, 0.5, 0.5], [2.0, 1.0, 0.7, 1.0], [6.0, 2.0, 0.3, 0.0]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return np.array(best[1], dtype=float)

def predict(row):
    return float(_basis(row, _params))