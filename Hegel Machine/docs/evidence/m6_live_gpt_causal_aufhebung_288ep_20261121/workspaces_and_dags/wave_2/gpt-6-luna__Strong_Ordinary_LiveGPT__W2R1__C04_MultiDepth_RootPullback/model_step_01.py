import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    af = float(row['a_foot'])
    impact = float(row['impact_idx'])
    indoor = ui / (1.0 + math.exp(6.0 * (af - 1.5)))
    effective = uo * pw + indoor
    loss_driver = (ui * impact) ** 2
    return effective, loss_driver

def fit(train_rows):
    e = np.array([_features(r)[0] for r in train_rows], dtype=float)
    q = np.array([_features(r)[1] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(theta):
        g, k, c = theta
        return g * e / (k + e) - c * q - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    g, k, c = fit(_TRAIN_ROWS)
    e, q = _features(row)
    return float(g * e / (k + e) - c * q)

_TRAIN_ROWS = []
