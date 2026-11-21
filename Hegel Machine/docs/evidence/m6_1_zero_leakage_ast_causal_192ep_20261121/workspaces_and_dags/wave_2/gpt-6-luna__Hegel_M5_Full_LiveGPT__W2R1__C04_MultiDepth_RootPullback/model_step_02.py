import numpy as np
from scipy.optimize import least_squares

def _features(row):
    dose = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    impact = float(row['impact_idx']) * float(row['a_foot'])
    return dose, impact

def fit(train_rows):
    ds = np.array([_features(r)[0] for r in train_rows], dtype=float)
    im = np.array([_features(r)[1] for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        gain, half, loss = p
        return gain * ds / (half + ds) - loss * im - ys
    result = least_squares(residual, x0=[8.0, 1.0, 5.0], bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    return result.x

def predict(row):
    gain, half, loss = fit(_TRAIN_ROWS)
    dose, impact = _features(row)
    return float(gain * dose / (half + dose) - loss * impact)

_TRAIN_ROWS = []