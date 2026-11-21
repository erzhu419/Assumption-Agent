import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    af = float(row['a_foot'])
    impact = float(row['impact_idx'])
    eindoor = ui / (1.0 + math.exp(6.0 * (af - 1.5)))
    eeff = uo * pw + eindoor
    loss_basis = (ui * impact) ** 2
    return eeff, loss_basis

def fit(train_rows):
    global _params
    xs = [_features(r) for r in train_rows]
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(theta):
        g, k, c = theta
        return np.asarray([g * (e / (k + e)) - c * loss for e, loss in xs]) - ys
    result = least_squares(residual, x0=np.array([6.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    g, k, c = _params
    e, loss = _features(row)
    return float(g * (e / (k + e)) - c * loss)
