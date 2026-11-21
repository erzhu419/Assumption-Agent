import numpy as np
from scipy.optimize import least_squares

def _features(row):
    dose = float(row['u_indoor']) + float(row['u_outdoor']) * float(row['p_weather'])
    impact = float(row['impact_idx']) * float(row['a_foot'])
    return dose, impact

def fit(train_rows):
    doses = np.array([_features(r)[0] for r in train_rows], dtype=float)
    impacts = np.array([_features(r)[1] for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residuals(params):
        gain, half_sat, loss = params
        pred = gain * doses / (half_sat + doses) - loss * impacts
        return pred - ys
    result = least_squares(residuals, x0=np.array([8.0, 1.0, 5.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    if not isinstance(row, dict):
        raise TypeError('predict expects a row dictionary')
    params = fit(_MODEL_TRAIN_ROWS)
    gain, half_sat, loss = params
    dose, impact = _features(row)
    return float(gain * dose / (half_sat + dose) - loss * impact)