import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    dose = outdoor + indoor
    impact = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(params):
        gain, half, penalty = params
        return gain * X[:, 0] / (half + X[:, 0]) - penalty * X[:, 1] - y
    result = least_squares(residual, x0=[6.0, 1.0, 0.5], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, half, penalty = fit(_TRAIN_ROWS)
    dose, impact = _features(row)
    return float(gain * dose / (half + dose) - penalty * impact)
