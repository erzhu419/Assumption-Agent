import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    delta = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    horizon = np.array([r['H'] for r in train_rows], dtype=float)
    shift = np.array([r['y'] - r['x_now'] for r in train_rows], dtype=float)

    def residual(params):
        steady, transient, rate = params
        return delta * (steady + transient * np.exp(-rate * horizon)) - shift

    _params = least_squares(residual, [20.0, 100.0, 1.0],
                            bounds=([-np.inf, -np.inf, 0.0], np.inf),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    steady, transient, rate = _params
    delta = row['u_now'] - row['u_past']
    return float(row['x_now'] + delta * (steady + transient * math.exp(-rate * row['H'])))