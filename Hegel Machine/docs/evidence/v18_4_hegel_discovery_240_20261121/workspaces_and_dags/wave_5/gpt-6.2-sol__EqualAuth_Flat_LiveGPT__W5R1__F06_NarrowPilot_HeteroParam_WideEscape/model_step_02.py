import numpy as np
from scipy.optimize import least_squares

_theta = None

def _basis(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    q = u / (1.2 + u)
    return (q, -q * s / (1.0 + s), np.sqrt(u))

def fit(train_rows):
    global _theta
    rows = list(train_rows)
    X = np.array([_basis(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _theta = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x

def predict(row) -> float:
    return float(np.dot(_basis(row), _theta))
