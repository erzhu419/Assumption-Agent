import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return u, s

def _law(row, p):
    j0, beta, rho, c = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(max(u, 0.0))

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_law(r, p) for r in train_rows]) - ys
    starts = [
        [j, b, rho, c]
        for j in (0.5, 2.0, 8.0, 20.0, 50.0)
        for b in (0.1, 1.0, 3.0, 8.0)
        for rho in (0.2, 0.7, 1.2)
        for c in (-2.0, 0.0, 2.0, 5.0)
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, 0.0, -5.0, -20.0], [1e4, 100.0, 5.0, 20.0]), max_nfev=3000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best[1].tolist()

def predict(row):
    p = fit(train_rows)
    return float(_law(row, p))