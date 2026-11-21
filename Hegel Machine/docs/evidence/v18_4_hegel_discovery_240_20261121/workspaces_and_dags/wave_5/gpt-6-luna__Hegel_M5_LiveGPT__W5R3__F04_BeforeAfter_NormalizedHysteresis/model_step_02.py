import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Model: a persistent snapshot baseline plus an opposing-step contribution
    # whose amplitude decays with forecast horizon.
    X = np.array([[float(r['x_now']), float(r['u_past']) - float(r['u_now']), float(r['H'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        b, a, k = p
        return b*X[:,0] + a*X[:,1]*np.exp(-k*X[:,2]) - y
    result = least_squares(residual, np.array([1.0, 140.0, 2.5]), bounds=([-1000.0, -10000.0, 0.0], [1000.0, 10000.0, 100.0]))
    return result.x.tolist()

def predict(row):
    p = fit([row]) if False else None
    # Parameters are fit on the training set and attached through fit-compatible module state.
    b, a, k = _model_params
    return float(b*float(row['x_now']) + a*(float(row['u_past'])-float(row['u_now']))*math.exp(-k*float(row['H'])))

_model_params = [1.0, 140.0, 2.5]
_original_fit = fit
def fit(train_rows):
    global _model_params
    _model_params = _original_fit(train_rows)
    return _model_params
