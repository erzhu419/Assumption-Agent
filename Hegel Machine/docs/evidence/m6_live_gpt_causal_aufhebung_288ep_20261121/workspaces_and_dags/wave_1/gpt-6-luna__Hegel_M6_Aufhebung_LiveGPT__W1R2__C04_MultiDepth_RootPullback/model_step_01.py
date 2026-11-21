import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    e_ind = row['u_indoor'] / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
    e_eff = row['u_outdoor'] * row['p_weather'] + e_ind
    loss = (row['u_indoor'] * row['impact_idx']) ** 2
    return e_eff, loss

def fit(train_rows):
    X = [_features(r) for r in train_rows]
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(theta):
        g, k, c = theta
        return np.array([g * (e / (k + e)) - c * loss for e, loss in X]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    g, k, c = _MODEL
    e, loss = _features(row)
    return float(g * (e / (k + e)) - c * loss)
