import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    a, b, c, k = p
    equilibrium = a * float(row['u_now']) + b
    decay = np.exp(-k * float(row['H']))
    return equilibrium + (c * float(row['x_now']) - equilibrium) * decay

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(p, row) - float(row['y']) for row in train_rows])
    candidates = []
    for k in (0.5, 1.0, 1.5, 2.0):
        result = least_squares(residual, [2.38, 0.0, 1.0, k], bounds=([-20.0, -20.0, -20.0, 0.01], [20.0, 20.0, 20.0, 5.0]), max_nfev=5000)
        candidates.append(result)
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(_params, row))