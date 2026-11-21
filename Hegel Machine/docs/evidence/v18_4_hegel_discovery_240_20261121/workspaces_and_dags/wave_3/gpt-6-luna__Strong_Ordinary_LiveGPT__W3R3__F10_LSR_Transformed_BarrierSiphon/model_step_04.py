import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, row['u_now'] / 0.02)
    s = row['s_grad']
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    ys = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    best = None
    for start in ([5.0, 0.4, 0.0], [10.0, 0.7, 1.0], [2.0, 0.1, 2.0], [20.0, 0.9, -1.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -20.0], [100.0, 1.0, 20.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return float(_value(row, fit([row])))