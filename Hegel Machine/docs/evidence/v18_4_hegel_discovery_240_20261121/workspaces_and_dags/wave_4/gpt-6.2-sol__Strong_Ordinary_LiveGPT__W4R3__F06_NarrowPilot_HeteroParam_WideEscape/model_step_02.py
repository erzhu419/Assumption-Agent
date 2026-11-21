import math
import numpy as np
from scipy.optimize import least_squares

def _basis(u, s):
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _coef
    X = np.asarray([_basis(float(r['u_now']), float(r['s_grad'])) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    _coef = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x
    return _coef

def predict(row):
    return float(np.dot(_basis(float(row['u_now']), float(row['s_grad'])), _coef))
