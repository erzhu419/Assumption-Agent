import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        k, a = p
        return np.array([k*r['u_now'] + (r['x_now'] - k*r['u_now'])*math.exp(-a*r['H']) - r['y'] for r in train_rows])
    result = least_squares(residual, [2.1, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = result.x

def predict(row):
    k, a = _params
    steady = k * row['u_now']
    return float(steady + (row['x_now'] - steady) * math.exp(-a * row['H']))