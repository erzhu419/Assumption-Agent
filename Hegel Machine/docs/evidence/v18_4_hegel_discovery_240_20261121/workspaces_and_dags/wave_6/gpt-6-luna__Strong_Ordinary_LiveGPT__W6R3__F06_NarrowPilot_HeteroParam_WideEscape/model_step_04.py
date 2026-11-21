import math
import numpy as np
from scipy.optimize import least_squares

# y = vmax * (1 - rho*s/(1+s)) * u/(1.2+u) + c_loc*sqrt(u)
def _features(row):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _value(row, p):
    vmax, rho, c_loc = p
    u, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
    return result.x.tolist()

def predict(row):
    return float(_value(row, _FITTED_PARAMS))
