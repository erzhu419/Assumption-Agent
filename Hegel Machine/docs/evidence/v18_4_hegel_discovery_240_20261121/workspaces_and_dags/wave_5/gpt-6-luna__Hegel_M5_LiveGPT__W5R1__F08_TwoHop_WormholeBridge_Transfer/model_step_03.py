import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # supplied characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(0.0, un))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(r, p) for r in train_rows]) - y
    # c_loc is unconstrained to resolve the documented open branch sign.
    best = None
    for x0 in ([6.0, 0.5, 0.0], [3.0, 0.8, 1.0], [8.0, 0.2, -1.0]):
        res = least_squares(residual, x0, bounds=([0.0, -5.0, -20.0], [100.0, 5.0, 20.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))
