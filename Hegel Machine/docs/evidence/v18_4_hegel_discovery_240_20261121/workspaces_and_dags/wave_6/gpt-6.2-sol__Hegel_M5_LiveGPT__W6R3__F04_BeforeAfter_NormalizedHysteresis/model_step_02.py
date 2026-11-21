import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'] - r['u_past'], r['H'], r['y'] - r['x_now']] for r in train_rows], dtype=float)
    du, h, dy = data.T

    def residual(p):
        steady, transient, rate = p
        return du * (steady + transient * np.exp(-rate * h)) - dy

    _params = least_squares(residual, [20.0, 130.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), x_scale='jac').x

def predict(row) -> float:
    steady, transient, rate = _params
    du = row['u_now'] - row['u_past']
    return float(row['x_now'] + du * (steady + transient * math.exp(-rate * row['H'])))
