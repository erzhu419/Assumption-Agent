import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return (base, -base * s / (1.0 + s), -u)

def fit(train_rows):
    global _params
    X = np.array([_features(row) for row in train_rows], dtype=float)
    y = np.array([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.ones(3), jac=lambda p: X).x

def predict(row) -> float:
    return float(np.dot(_features(row), _params))
