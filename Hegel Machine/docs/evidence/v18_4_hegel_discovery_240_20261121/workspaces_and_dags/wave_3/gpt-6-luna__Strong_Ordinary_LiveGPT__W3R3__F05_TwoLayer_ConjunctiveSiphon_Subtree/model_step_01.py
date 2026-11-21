import math
import numpy as np
from scipy.optimize import least_squares

def _basis(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return (u / (1.2 + u), s / (1.0 + s), math.sqrt(max(0.0, u)))

def fit(train_rows):
    X = np.array([_basis(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # y = vmax*(1-rho*s/(1+s))*u/(1.2+u) + c_loc*sqrt(u)
    def residual(p):
        vmax, rho, c_loc = p
        return vmax * X[:, 0] * (1.0 - rho * X[:, 1]) + c_loc * X[:, 2] - y
    best = None
    for p0 in ([4.0, 0.5, 0.0], [6.0, 0.8, 1.0], [3.0, 0.2, -1.0]):
        result = least_squares(residual, p0, bounds=([0.0, 0.0, -20.0], [100.0, 2.0, 20.0]))
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return tuple(float(v) for v in best[1])

def predict(row):
    vmax, rho, c_loc = fit(_FIT_ROWS)
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(0.0, u)))
