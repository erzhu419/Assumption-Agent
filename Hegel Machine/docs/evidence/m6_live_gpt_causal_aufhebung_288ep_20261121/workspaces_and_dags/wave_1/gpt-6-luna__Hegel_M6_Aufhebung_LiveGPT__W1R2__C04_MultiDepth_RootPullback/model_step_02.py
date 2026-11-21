import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    af = float(row['a_foot'])
    impact = float(row['impact_idx'])
    e_indoor = ui / (1.0 + math.exp(6.0 * (af - 1.5)))
    e_eff = uo * pw + e_indoor
    loss = (ui * impact) ** 2
    return e_eff, loss

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    X = [_features(row) for row in rows]
    y = np.asarray([float(row['y']) for row in rows], dtype=float)
    def residual(params):
        g, k, c = params
        return np.asarray([g * (e / (k + e)) - c * loss for e, loss in X]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.1]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _MODEL = tuple(float(v) for v in result.x)
    return _MODEL

def predict(row):
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    g, k, c = _MODEL
    e, loss = _features(row)
    return float(g * (e / (k + e)) - c * loss)