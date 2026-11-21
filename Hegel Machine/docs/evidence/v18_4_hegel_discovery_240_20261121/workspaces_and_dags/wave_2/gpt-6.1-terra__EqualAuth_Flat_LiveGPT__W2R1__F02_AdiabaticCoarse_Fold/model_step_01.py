import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        k, log_a = p
        a = math.exp(log_a)
        return np.array([k*r['u_now'] + (r['x_now'] - k*r['u_now'])*math.exp(-a*r['H']) - r['y'] for r in train_rows])
    result = least_squares(residual, [1.83, 0.0], bounds=([-100.0, -10.0], [100.0, 10.0]))
    _params = result.x

def predict(row):
    k, log_a = _params
    return float(k*row['u_now'] + (row['x_now'] - k*row['u_now'])*math.exp(-math.exp(log_a)*row['H']))