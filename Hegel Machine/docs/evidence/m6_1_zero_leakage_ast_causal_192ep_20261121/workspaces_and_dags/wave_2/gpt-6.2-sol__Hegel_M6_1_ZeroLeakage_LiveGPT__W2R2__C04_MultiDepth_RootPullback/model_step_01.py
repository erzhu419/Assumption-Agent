import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    gate = 1.0 / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * gate
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residuals(params):
        gain, half_sat, impact_coef = params
        return gain * dose / (half_sat + dose) - impact_coef * impact - y

    _params = least_squares(residuals, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    dose, impact = _features(row)
    gain, half_sat, impact_coef = _params
    return float(gain * dose / (half_sat + dose) - impact_coef * impact)
