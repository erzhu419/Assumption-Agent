import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    indoor = row['u_indoor'] / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
    eff = row['u_outdoor'] * row['p_weather'] + indoor
    loss = (row['u_indoor'] * row['impact_idx']) ** 2
    return eff, loss

def fit(train_rows):
    rows = list(train_rows)
    x = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(theta):
        g, k, c = theta
        return g * x[:, 0] / (k + x[:, 0]) - c * x[:, 1] - y
    result = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    g, k, c = fit(_TRAIN_ROWS)
    eff, loss = _features(row)
    return float(g * eff / (k + eff) - c * loss)
