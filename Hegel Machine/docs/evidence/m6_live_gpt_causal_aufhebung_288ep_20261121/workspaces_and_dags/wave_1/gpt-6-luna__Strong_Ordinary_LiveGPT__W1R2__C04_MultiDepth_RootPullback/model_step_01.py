import math
import numpy as np
from scipy.optimize import least_squares

def _dose(row):
    outdoor = float(row.get('u_outdoor', 0.0)) * float(row.get('p_weather', 0.0))
    indoor = float(row.get('u_indoor', 0.0)) / (1.0 + math.exp(6.0 * (float(row.get('a_foot', 1.5)) - 1.5)))
    return outdoor + indoor

def _features(row):
    e = _dose(row)
    saturating = e / (1.0 + e)  # replaced below by fitted k_e
    loss_shape = (float(row.get('u_indoor', 0.0)) * float(row.get('impact_idx', 0.0))) ** 2
    return e, loss_shape

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    X = np.asarray([_features(r) for r in train_rows], dtype=float)
    def residual(theta):
        g, k, c = theta
        return g * X[:, 0] / (k + X[:, 0]) - c * X[:, 1] - ys
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    g, k, c = result.x
    def predict(row):
        e, loss_shape = _features(row)
        return float(g * e / (k + e) - c * loss_shape)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first to estimate the mechanism parameters.')