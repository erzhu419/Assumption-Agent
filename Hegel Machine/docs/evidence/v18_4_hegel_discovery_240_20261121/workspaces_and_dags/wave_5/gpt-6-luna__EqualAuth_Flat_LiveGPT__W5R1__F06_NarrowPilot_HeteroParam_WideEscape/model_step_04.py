import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, km, a, b, c = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * u / (km + u) * (a + b / (s + c))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_pred(p, r) for r in train_rows]) - y
    starts = [np.array([5.0, 1.0, 0.3, 1.0, 0.2]), np.array([7.0, 2.0, 0.5, 1.5, 0.5])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([1e-6, 1e-6, 1e-6, 1e-8, 1e-6], [100.0, 100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    raise ValueError('predict(row) requires fitted parameters; call fit(train_rows) and use the fitted parameters in prediction.')