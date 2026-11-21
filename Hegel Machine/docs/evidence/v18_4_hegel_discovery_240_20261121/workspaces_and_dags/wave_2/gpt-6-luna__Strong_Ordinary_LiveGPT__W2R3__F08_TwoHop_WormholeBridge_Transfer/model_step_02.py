import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 0.3, 0.5]), max_nfev=20000)
    return tuple(float(v) for v in result.x)

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(row, p))