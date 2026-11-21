import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    gate = 1.0 / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * gate
    impact_sq = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact_sq

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact_sq = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residuals(params):
        gain, half_sat, impact_coef = params
        return gain * dose / (half_sat + dose) - impact_coef * impact_sq - y

    result = least_squares(residuals, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], np.inf))
    _params = result.x
    return _params

def predict(row) -> float:
    dose, impact_sq = _features(row)
    gain, half_sat, impact_coef = _params
    return float(gain * dose / (half_sat + dose) - impact_coef * impact_sq)
