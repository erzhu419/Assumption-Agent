import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    best = None
    for x0 in ([4.0, 0.6, 0.5], [8.0, 0.3, 0.1], [3.0, 0.8, 1.0]):
        result = least_squares(residual, x0=x0, bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]), max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    p = fit(_CURRENT_TRAIN_ROWS)
    return float(_pred(row, p))