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
    result = least_squares(residual, [1.25, 0.4], bounds=([-10.0, -5.0], [10.0, 5.0]))
    _params = result.x

def predict(row):
    k, log_a = _params
    steady = k*row['u_now']
    return float(steady + (row['x_now'] - steady)*math.exp(-math.exp(log_a)*row['H']))
