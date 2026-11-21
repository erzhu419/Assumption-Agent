import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    weather = float(row['p_weather'])
    area = float(row['a_foot'])
    impact = float(row['impact_idx'])
    indoor = ui / (1.0 + math.exp(max(-700.0, min(700.0, 6.0 * (area - 1.5)))))
    dose = uo * weather + indoor
    loss_basis = (ui * impact) ** 2
    return dose, loss_basis

def fit(train_rows):
    global _PARAMS
    doses = np.array([_features(r)[0] for r in train_rows], dtype=float)
    losses = np.array([_features(r)[1] for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(theta):
        gain, half, penalty = theta
        return gain * doses / (half + doses) - penalty * losses - ys
    result = least_squares(residual, x0=np.array([6.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], ([100.0, 100.0, 100.0])))
    _PARAMS = tuple(float(x) for x in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    gain, half, penalty = _PARAMS
    dose, loss_basis = _features(row)
    return float(gain * dose / (half + dose) - penalty * loss_basis)
