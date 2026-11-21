import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        k, a = p
        return np.array([k*r['u_now'] + (r['x_now'] - k*r['u_now'])*math.exp(-a*r['H']) - r['y'] for r in train_rows])
    result = least_squares(residual, [1.86, 1.2], bounds=([-100.0, 0.0], [100.0, 20.0]), max_nfev=10000)
    _params = result.x

def predict(row):
    k, a = _params
    equilibrium = k * row['u_now']
    return float(equilibrium + (row['x_now'] - equilibrium) * math.exp(-a * row['H']))