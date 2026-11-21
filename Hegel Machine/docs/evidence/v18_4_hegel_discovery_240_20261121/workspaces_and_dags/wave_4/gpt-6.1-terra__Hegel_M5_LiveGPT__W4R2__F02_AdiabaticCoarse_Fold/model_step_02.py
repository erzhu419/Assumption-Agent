import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        equilibrium = p[0] * u
        return equilibrium + (x - equilibrium) * np.exp(-p[1] * h) - y
    _params = least_squares(residual, [2.2, 1.4], bounds=([0.0, 0.0], [np.inf, np.inf])).x

def predict(row):
    equilibrium = _params[0] * float(row['u_now'])
    return float(equilibrium + (float(row['x_now']) - equilibrium) * np.exp(-_params[1] * float(row['H'])))