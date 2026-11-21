import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = ([10.0, 0.5, 0.0], [15.0, 0.8, -1.0], [8.0, 0.2, 1.0])
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, -20.0], [100.0, 2.0, 20.0]), max_nfev=10000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    return tuple(float(v) for v in best.x)

def predict(row):
    return float(_value(row, _PARAMS))