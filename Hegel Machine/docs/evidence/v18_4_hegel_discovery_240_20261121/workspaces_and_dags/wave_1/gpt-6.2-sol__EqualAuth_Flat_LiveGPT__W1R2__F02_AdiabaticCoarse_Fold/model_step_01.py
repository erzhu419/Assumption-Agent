import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        return k*u + (x-k*u)*np.exp(-a*h) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    _params = least_squares(residual, [k0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row):
    k, a = _params
    x_ss = k * float(row['u_now'])
    return float(x_ss + (float(row['x_now']) - x_ss)*math.exp(-a*float(row['H'])))