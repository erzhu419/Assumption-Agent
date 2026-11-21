import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u_out = float(row['u_outdoor'])
    u_in = float(row['u_indoor'])
    p = float(row['p_weather'])
    a = float(row['a_foot'])
    impact = float(row['impact_idx'])
    space_factor = 1.0 / (1.0 + math.exp(6.0 * (a - 1.5)))
    dose = u_out * p + u_in * space_factor
    penalty_basis = (u_in * impact) ** 2
    return dose, penalty_basis

def fit(train_rows):
    global _params
    doses = np.array([_features(r)[0] for r in train_rows], dtype=float)
    penalties = np.array([_features(r)[1] for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)

    def residual(params):
        g, k, c = params
        return g * doses / (k + doses) - c * penalties - ys

    result = least_squares(residual, x0=np.array([8.0, 1.0, 1.0]), bounds=([0.0, 1e-9, 0.0], ([np.inf, np.inf, np.inf])))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    g, k, c = _params
    dose, penalty_basis = _features(row)
    return float(g * dose / (k + dose) - c * penalty_basis)
