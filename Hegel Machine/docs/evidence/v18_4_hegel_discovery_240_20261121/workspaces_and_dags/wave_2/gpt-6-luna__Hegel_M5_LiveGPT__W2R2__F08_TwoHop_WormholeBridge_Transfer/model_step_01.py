import math
import numpy as np
from scipy.optimize import least_squares

def _basis(row):
    u = float(row['u_now'])
    return np.array([u, -(0.2*u)**4], dtype=float)

def fit(train_rows):
    X = np.vstack([_basis(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Fit the two coefficients in the verified convective-minus-radiative law.
    res = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 0.01]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_basis(row) @ p)
