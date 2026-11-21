import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    k_ss, a_eff = p
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return k_ss * u + (x - k_ss * u) * math.exp(-a_eff * H)

def fit(train_rows):
    def residual(p):
        return np.array([_pred(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([1.25, 1.0]), max_nfev=10000)
    return tuple(float(v) for v in result.x)

def predict(row):
    return float(_pred(row, fit([dict(row, y=0.0)])))