import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    un = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    base = (un / (1.2 + un))
    atten = 1.0 - (s / (1.0 + s))
    return un, s, base, atten

def fit(train_rows):
    # Fit the source-inland-hub saturation/gradient law with a signed sqrt-input branch.
    def pred(p, row):
        un, s, base, atten = _features(row)
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * base + c_loc * math.sqrt(max(un, 0.0))
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.0]), max_nfev=3000)
    return result.x

def predict(row):
    p = fit(_ACTIVE_TRAIN_ROWS)
    un, s, base, atten = _features(row)
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * base + c_loc * math.sqrt(max(un, 0.0)))