import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, r):
        a, b, k, d = p
        return a + b * (1.0 - math.exp(-k * float(r['u_now']))) + d * (float(r['s_grad']) - 1.0)
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.0, 3.0, 0.7, -0.7]), max_nfev=20000)
    return result.x

def predict(row):
    a, b, k, d = _FIT_PARAMS
    return a + b * (1.0 - math.exp(-k * float(row['u_now']))) + d * (float(row['s_grad']) - 1.0)
