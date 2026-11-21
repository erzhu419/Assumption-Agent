import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    tau = 1.0
    un = u / tau
    return un, s

def _value(row, p):
    vmax, rho, c_loc = p
    un, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(un)

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows])
    x0 = np.array([6.5, 0.70, 0.1])
    lo = np.array([1.0, 0.05, -2.0])
    hi = np.array([15.0, 0.95, 2.0])
    result = least_squares(lambda p: np.asarray([_value(r, p) for r in train_rows]) - ys,
                           x0, bounds=(lo, hi), max_nfev=10000)
    return result.x.tolist()

def predict(row):
    return float(_value(row, fit([dict(row, y=0.0)])))