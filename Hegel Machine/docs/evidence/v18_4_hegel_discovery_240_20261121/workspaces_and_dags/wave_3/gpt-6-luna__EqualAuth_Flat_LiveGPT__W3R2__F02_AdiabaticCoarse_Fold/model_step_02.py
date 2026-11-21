import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        E, eta = np.exp(p)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.57, 1.0]))
    E, eta = np.exp(result.x)
    return {'E': float(E), 'eta': float(eta)}

def predict(row):
    E, eta = _MODEL['E'], _MODEL['eta']
    return float((row['u_now'] / E) * (-math.expm1(-E * row['H'] / eta)))

_MODEL = {'E': 1.0, 'eta': 1.0}
_original_fit = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _original_fit(train_rows)
    return _MODEL
