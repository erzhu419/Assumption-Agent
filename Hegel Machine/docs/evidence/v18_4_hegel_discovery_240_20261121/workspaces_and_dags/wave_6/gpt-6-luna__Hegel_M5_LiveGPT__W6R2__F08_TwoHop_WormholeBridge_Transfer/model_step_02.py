import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _theta
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, ss):
        a, b, k, c, d = p
        return a + b * (1.0 - np.exp(-k * uu)) + (ss - 1.0) * (c + d * uu)
    def residual(p):
        return pred(p, u, s) - y
    result = least_squares(residual, x0=np.array([0.4, 2.2, 0.8, 0.0, 0.0]), bounds=([-20, -20, 1e-6, -20, -20], [20, 20, 20, 20, 20]))
    _theta = result.x
    return _theta

def predict(row):
    a, b, k, c, d = _theta
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a + b * (1.0 - math.exp(-k * u)) + (s - 1.0) * (c + d * u))