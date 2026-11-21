import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(r, p) for r in rows]) - y
    # Multiple feasible starts reduce sensitivity to the threshold's piecewise kink.
    best = None
    for start in ([5.0, 0.7, 0.0, 1.0, 2.6], [8.0, 0.5, -0.2, 0.8, 2.5], [4.0, 0.9, 0.3, 1.5, 2.68]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.5, 10.0, 20.0, 3.2]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict expects fitted parameters through row["train_rows"]')
    return float(_pred(row, p))