import numpy as np
from scipy.optimize import least_squares

def _terms(u, s):
    f = u / (1.2 + u)
    return np.array([f, -f * s / (1.0 + s), np.sqrt(u)])

def fit(train_rows):
    global _params
    X = np.array([_terms(r['u_now'], r['s_grad']) for r in train_rows])
    y = np.array([r['y'] for r in train_rows])
    _params = least_squares(lambda p: X @ p - y, [7.0, 1.0, 0.25]).x
    return _params

def predict(row):
    return float(_terms(row['u_now'], row['s_grad']) @ _params)
