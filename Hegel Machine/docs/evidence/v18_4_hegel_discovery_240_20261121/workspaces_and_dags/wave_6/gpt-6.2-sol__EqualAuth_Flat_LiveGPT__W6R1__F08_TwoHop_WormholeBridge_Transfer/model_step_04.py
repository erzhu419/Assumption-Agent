import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a, b, c, d, k = p
        return a*u/((1+b*u)*(1+c*s)) - d*np.expm1(-k*u) - y

    _params = least_squares(residual, [4.5, 0.8, 1.0, 1.0, 1.0], bounds=(0.0, np.inf)).x

def predict(row):
    a, b, c, d, k = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(a*u/((1+b*u)*(1+c*s)) - d*math.expm1(-k*u))